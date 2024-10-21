import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def compressibility():
    config = base.get_config()
    config.project_name = "compressibility"

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"

    config.num_epochs = 50
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    # this corresponds to (8 * 4) / (4 * 2) = 4 gradient updates per epoch.
    config.train.batch_size = 2
    config.train.gradient_accumulation_steps = 4

    # prompting
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "jpeg_compressibility"
    config.intrinsic_reward_fn = "baseline"
    config.intrinsic_reward_weight = 0.0

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }

    return config


def compressibility_ablation():
    config = compressibility()
    config.intrinsic_reward_fn = "intrinsic"
    config.intrinsic_reward_weight = 0.009  # 0.009 > 0.008 > 0.01
    return config


def compressibility_ada():
    config = compressibility()
    config.intrinsic_reward_fn = "intrinsic_ada"
    config.intrinsic_reward_weight = 0.009  # 0.009 > 0.008 > 0.01
    return config


def incompressibility():
    config = compressibility()
    config.project_name = "incompressibility"
    config.reward_fn = "jpeg_incompressibility"
    config.intrinsic_reward_fn = "baseline"
    config.intrinsic_reward_weight = 0.0
    return config


def incompressibility_ablation():
    config = incompressibility()
    config.intrinsic_reward_fn = "intrinsic"
    config.intrinsic_reward_weight = 0.015
    return config


def incompressibility_ada():
    config = incompressibility()
    config.intrinsic_reward_fn = "intrinsic_ada"
    config.intrinsic_reward_weight = 0.015  # 0.01 > 0.009
    return config


def aesthetic():
    config = compressibility()
    config.project_name = ""
    config.num_epochs = 100
    config.reward_fn = "aesthetic_score"
    config.intrinsic_reward_fn = "baseline"
    config.intrinsic_reward_weight = 0.0

    # this reward is a bit harder to optimize, so I used 2 gradient updates per epoch.
    config.train.batch_size = 1
    config.train.gradient_accumulation_steps = 8

    config.prompt_fn = "simple_animals"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }
    return config


def aesthetic2():
    config = aesthetic()
    config.intrinsic_reward_fn = "baseline"
    config.intrinsic_reward_weight = 0.0
    return config


def intrinsic_only():
    config = aesthetic()
    config.reward_fn = "dummy_aesthetic_score"
    config.intrinsic_reward_fn = "intrinsic"
    config.intrinsic_reward_weight = 0.025
    return config


def new_baseline_aesthetic():  # 只给最后一步加外部奖励、其他步加内部奖励效果不好
    config = aesthetic()
    config.reward_fn = "extrinsic_aesthetic_score"
    config.intrinsic_reward_fn = "intrinsic"
    config.intrinsic_reward_weight = 0.009
    return config


def aesthetic_ablation():
    config = aesthetic2()
    config.intrinsic_reward_fn = "intrinsic"
    config.intrinsic_reward_weight = 0.005
    return config


def aesthetic_ada():
    config = aesthetic2()
    config.intrinsic_reward_fn = "intrinsic_ada"
    config.intrinsic_reward_weight = 0.005
    return config


def prompt_image_alignment():
    config = compressibility()

    config.num_epochs = 200
    # for this experiment, I reserved 2 GPUs for LLaVA inference so only 6 could be used for DDPO. the total number of
    # samples per epoch is 8 * 6 * 6 = 288.
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 6

    # again, this one is harder to optimize, so I used (8 * 6) / (4 * 6) = 2 gradient updates per epoch.
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 6

    # prompting
    config.prompt_fn = "nouns_activities"
    config.prompt_fn_kwargs = {
        "nouns_file": "simple_animals.txt",
        "activities_file": "activities.txt",
    }

    # rewards
    config.reward_fn = "llava_bertscore"

    config.per_prompt_stat_tracking = {
        "buffer_size": 32,
        "min_count": 16,
    }

    return config


def get_config(name):
    return globals()[name]()
