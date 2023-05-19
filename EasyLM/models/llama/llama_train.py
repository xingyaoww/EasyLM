import pprint
from functools import partial

from tqdm import tqdm, trange
import numpy as np
import mlxu

import jax
import flax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
import optax
from typing import Any, Mapping, Text, Tuple, Union, NamedTuple, Optional

from jax_smi import initialise_tracking
from jax.experimental.compilation_cache import compilation_cache as cc
initialise_tracking()
cc.initialize_cache("/tmp/jax_cache")

from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint,
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLMModule
)


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    num_microbatches=1,
    initialize_jax_distributed=False,
    mesh_dim='1,-1,1',
    dtype='fp32',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
)


def main(argv):
    if FLAGS.initialize_jax_distributed:
        jax.distributed.initialize()

    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = mlxu.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != '':
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length

    if FLAGS.load_llama_config != '':
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
    else:
        llama_config = LLaMAConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(
        bos_token_id=dataset.tokenizer.bos_token_id,
        eos_token_id=dataset.tokenizer.eos_token_id,
    ))
    if llama_config.vocab_size < dataset.vocab_size:
        llama_config.update(dict(vocab_size=dataset.vocab_size))

    model = FlaxLLaMAForCausalLMModule(
        llama_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(LLaMAConfig.get_weight_decay_exclusions())
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(llama_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step_microbatched(
        train_state, rng, batch,
        num_microbatches: int = 1,
    ):
        """Implements optional microbatched gradient accumulation.

        Args:
        loss_fn: The loss function that takes in (train_state.params, batch, dropout_rng).
        train_state: A train state with model parameters and optimizer state.
        batch: A batch of data.
        dropout_rng: jax PRNGKey for dropout.
        num_microbatches: the number of microbatches to use, or None for direct
            training.
        data_partition_spec: the PartitionSpec to use for partitioning annotations
            on the batch.

        Returns:
        Accumulated gradients and incremental metrics.
        """
        batch_size = batch['input_tokens'].shape[0]
        microbatch_size = batch_size // num_microbatches
        accum_dtype = jnp.float32

        def loss_and_accuracy(params, batch, rng):
            batch = with_sharding_constraint(
                batch, PS(('dp', 'fsdp'))
            )
            logits = model.apply(
                params, batch['input_tokens'], deterministic=False,
                rngs=rng,
            ).logits
            loss, accuracy = cross_entropy_loss_and_accuracy(
                logits, batch['target_tokens'], batch['loss_masks']
            )
            return loss, {'accuracy': accuracy}

        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)

        def get_microbatch(batch: dict, idx: int) -> Mapping[str, jnp.ndarray]:
            """Fetch microbatch slice from possibly-packed input data."""
            offset = idx * microbatch_size
            length = microbatch_size
            starts = {k: [offset] + [0] * (b.ndim - 1)
                    for k, b in batch.items()}
            limits = {k: [length] + list(b.shape[1:])
                    for k, b in batch.items()}
            return {
                k: jax.lax.dynamic_slice(b, starts[k], limits[k])
                for k, b in batch.items()
            }

        def calculate_grad(loop_cnt, rng):
            mbatch = get_microbatch(batch, loop_cnt)
            # We need to annotate the microbatch sharding as we would a batch.
            mbatch = jax.tree_util.tree_map(
                lambda x: with_sharding_constraint(x, PS('dp')),
                mbatch
            )
            (loss, metrics), grad = grad_fn(
                train_state.params,
                mbatch,
                rng,
            )
            return loss, grad, metrics

        def per_microbatch_train_step(
            loop_cnt: int, state: Tuple[jnp.ndarray, jnp.ndarray,
                                        Mapping[str, jnp.ndarray],
                                        Optional[flax.core.FrozenDict]]
        ) -> Tuple[jnp.ndarray, jnp.ndarray, Mapping[str, jnp.ndarray],
                Optional[flax.core.FrozenDict]]:
            (rng, loss_accum, grad_accum, metrics_accum) = state
            loss, grad, metrics = calculate_grad(loop_cnt, rng)
            
            # convert to accum_dtype
            loss = loss.astype(accum_dtype)
            grad = jax.tree_util.tree_map(
                lambda x: x.astype(accum_dtype), grad
            )
            metrics = jax.tree_util.tree_map(
                lambda x: x.astype(accum_dtype), metrics
            )

            loss_accum = loss_accum + loss
            metrics_accum = jax.tree_util.tree_map(
                jnp.add, metrics_accum, metrics
            )
            grad_accum = jax.tree_util.tree_map(jnp.add, grad_accum, grad)
            return rng, loss_accum, grad_accum, metrics_accum

        # Initialize gradient accumulation loop state.
        loss_accum_init = jnp.zeros((), accum_dtype)
        grad_accum_init = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape, accum_dtype),
            train_state.params
        )

        rng_generator = JaxRNG(rng)
        input_rng = rng_generator(llama_config.rng_keys())
        _, _, initial_metrics_shape = jax.eval_shape(
            calculate_grad, loop_cnt=0,
            rng=input_rng
        )

        metrics_accum_init = {
            k: jnp.zeros((), accum_dtype)
            for k in initial_metrics_shape
        }
        loop_init = (
            input_rng, # same rng for all microbatches
            loss_accum_init,
            grad_accum_init,
            metrics_accum_init
        )
        _, loss_accum, grad_accum, metrics_accum = jax.lax.fori_loop(
            0, num_microbatches, per_microbatch_train_step, loop_init
        )

        # Apply the gradients to the model.
        train_state = train_state.apply_gradients(grads=grad_accum)
        metrics = dict(
            loss=loss_accum / num_microbatches,
            accuracy=metrics_accum['accuracy'] / num_microbatches,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            gradient_norm=global_norm(grad_accum),
            param_norm=global_norm(train_state.params),
        )
        new_rng = rng_generator()
        return train_state, new_rng, metrics

    def eval_step(train_state, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        logits = model.apply(
            train_state.params, batch['input_tokens'], deterministic=True,
            rngs=rng_generator(llama_config.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
        )
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        LLaMAConfig.get_partition_rules(), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    print(f"Number of microbatches: {FLAGS.num_microbatches}")
    sharded_train_step = pjit(
        partial(train_step_microbatched, num_microbatches=FLAGS.num_microbatches),
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            dataset=dataset.get_state_dict(),
            milestone=milestone,
        )

    mesh = LLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(0, FLAGS.total_steps, ncols=0)

        for step, (batch, dataset_metrics) in zip(step_counter, dataset):
            # Skip steps until start_step
            if step < start_step:
                continue

            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, batch
            )

            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, eval_batch
                        )
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step}
                log_metrics.update(metrics)
                log_metrics.update(dataset_metrics)
                log_metrics = jax.device_get(log_metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    mlxu.run(main)
