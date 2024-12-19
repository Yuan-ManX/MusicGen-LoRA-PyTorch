import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import datasets
import numpy as np
import torch
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForTextToWaveform,
    AutoModel,
    AutoProcessor,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers.integrations import is_wandb_available
from multiprocess import set_start_method


# 检查 Transformers 的最小版本是否为 "4.40.0.dev0"
check_min_version("4.40.0.dev0")
# 检查 Datasets 的最小版本是否为 "2.12.0"
require_version("datasets>=2.12.0")


# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    """
    创建一个返回列表的字段，默认值为 `default`，并附加 `metadata`。

    Args:
        default (list, optional): 默认的列表值。默认为 None。
        metadata (dict, optional): 附加的元数据信息。默认为 None。

    Returns:
        dataclasses.field: 配置好的数据类字段。
    """
    # 使用 lambda 函数作为 default_factory，确保每次实例化时都返回一个新的列表
    return field(default_factory=lambda: default, metadata=metadata)


class MusicgenTrainer(Seq2SeqTrainer):
    def _pad_tensors_to_max_len(self, tensor, max_length):
        """
        将张量填充到指定的最大长度。

        如果张量的长度小于最大长度，则在末尾填充指定的填充标记（pad token）。
        如果张量的长度大于或等于最大长度，则截断到最大长度。

        Args:
            tensor (torch.Tensor): 需要填充的张量，形状应为 [batch_size, seq_length, ...]。
            max_length (int): 填充后的最大长度。

        Returns:
            torch.Tensor: 填充后的张量，形状为 [batch_size, max_length, ...]。

        Raises:
            ValueError: 如果模型配置中没有设置 pad_token_id。
        """
        # 检查 tokenizer 是否存在并且具有 pad_token_id 属性
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            # 如果 tokenizer 有 pad_token_id，则使用它
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                # 如果没有 pad_token_id，则使用 eos_token_id 作为备选
                else self.tokenizer.eos_token_id 
            )
        else:
            # 如果 tokenizer 不存在或没有 pad_token_id，则检查模型配置中是否有 pad_token_id
            if self.model.config.pad_token_id is not None:
                # 使用模型配置中的 pad_token_id
                pad_token_id = self.model.config.pad_token_id
            else:
                # 如果模型配置中也没有设置 pad_token_id，则抛出错误
                raise ValueError(
                    "Pad_token_id must be set in the configuration of the model, in order to pad tensors"
                )

        # 创建一个填充后的张量，形状为 [batch_size, max_length, tensor.shape[2]]
        # 填充值为 pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length, tensor.shape[2]),
            dtype=tensor.dtype,
            device=tensor.device,
        )

        # 计算需要填充的长度，确保不超过最大长度
        length = min(max_length, tensor.shape[1])
        # 将原始张量的前 `length` 个时间步赋值给填充后的张量
        padded_tensor[:, :length] = tensor[:, :length]
        # 返回填充后的张量
        return padded_tensor


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    """
    与我们将要微调的具体模型配置/分词器相关的参数。
    """

    # 预训练模型的路径或模型标识符。可以是本地路径或 Hugging Face 模型库中的名称。
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )

    # 预训练配置名称或路径。如果未指定，则默认使用 model_name 对应的配置。
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )

    # 预训练处理器名称或路径。如果未指定，则默认使用 model_name 对应的处理器。
    processor_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained processor name or path if not the same as model_name"
        },
    )

    # 预训练模型下载后存储的缓存目录。如果未指定，则使用默认缓存目录。
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )

    # 是否使用快速分词器（基于 tokenizers 库）。默认为 True。
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )

    # 要使用的模型版本。可以是分支名称、标签名称或提交ID。默认为 "main"。
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )

    # 填充标记的ID。如果指定，则更改模型的填充标记ID。
    pad_token_id: int = field(
        default=None,
        metadata={"help": "If specified, change the model pad token id."},
    )

    # 解码器起始标记的ID。如果指定，则更改模型的解码器起始标记ID。
    decoder_start_token_id: int = field(
        default=None,
        metadata={"help": "If specified, change the model decoder start token id."},
    )

    # 是否冻结文本编码器。如果为 True，则在训练过程中不更新文本编码器的参数。默认为 True。
    freeze_text_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the text encoder."},
    )

    # 用于在评估期间计算音频相似度的模型路径或标识符。默认为 "laion/larger_clap_music_and_speech"。
    clap_model_name_or_path: str = field(
        default="laion/larger_clap_music_and_speech",
        metadata={
            "help": "Used to compute audio similarity during evaluation. Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    # 是否使用 Lora 技术进行模型微调。默认为 False。
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use Lora."},
    )

    # 模型的指导尺度。如果指定，则更改模型的指导尺度。
    guidance_scale: float = field(
        default=None,
        metadata={"help": "If specified, change the model guidance scale."},
    )


@dataclass
class DataSeq2SeqTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command line.
    """
    """
    与我们将输入模型进行训练和评估的数据相关的参数。

    使用 `HfArgumentParser` 可以将这个类转换为命令行参数，以便在命令行中指定它们。
    """

    # 要使用的数据集的配置名称（通过 datasets 库）。
    dataset_name: str = field(
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        }
    )

    # 数据集的配置名称（通过 datasets 库）。如果未指定，则使用默认配置。
    dataset_config_name: str = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )

    # 要使用的训练数据集分割的名称（通过 datasets 库）。默认为 "train+validation"。
    train_split_name: str = field(
        default="train+validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to "
                "'train+validation'"
            )
        },
    )

    # 要使用的评估数据集分割的名称（通过 datasets 库）。默认为 'test'。
    eval_split_name: str = field(
        default="test",
        metadata={
            "help": "The name of the evaluation data set split to use (via the datasets library). Defaults to 'test'"
        },
    )

    # 包含目标音频数据的列名。默认为 'audio'。
    target_audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the target audio data. Defaults to 'audio'"
        },
    )

    # 如果设置，则为包含文本数据的描述列的名称。如果未设置，应将 `add_metadata` 设置为 True，以自动生成音乐描述。
    text_column_name: str = field(
        default=None,
        metadata={
            "help": "If set, the name of the description column containing the text data. If not, you should set `add_metadata` to True, to automatically generates music descriptions ."
        },
    )

    # 如果设置并且 `add_metadata=True`，则将实例提示添加到音乐描述中。这允许使用此实例提示作为锚点，让模型学习将其与数据集的特定性相关联。
    instance_prompt: str = field(
        default=None,
        metadata={
            "help": "If set and `add_metadata=True`, will add the instance prompt to the music description. For example, if you set this to `punk`, `punk` will be added to the descriptions. This allows to use this instance prompt as an anchor for your model to learn to associate it to the specificities of your dataset."
        },
    )

    # 如果设置，则为包含条件音频数据的列名。这是完全可选的，仅用于条件引导生成。
    conditional_audio_column_name: str = field(
        default=None,
        metadata={
            "help": "If set, the name of the dataset column containing conditional audio data. This is entirely optional and only used for conditional guided generation."
        },
    )

    # 是否覆盖缓存的预处理数据集。
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )

    # 用于预处理的进程数。
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # 出于调试目的或加快训练速度，如果设置，则将训练示例的数量截断为此值。
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )

    # 出于调试目的或加快训练速度，如果设置，则将验证示例的数量截断为此值。
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of validation examples to this "
                "value if set."
            )
        },
    )

    # 过滤超过 `max_duration_in_seconds` 秒的音频文件。
    max_duration_in_seconds: float = field(
        default=30.0,
        metadata={
            "help": (
                "Filter audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )

    # 过滤少于 `min_duration_in_seconds` 秒的音频文件。
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={
            "help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"
        },
    )

    # 在评估期间将使用此提示作为额外的生成样本。
    full_generation_sample_text: str = field(
        default="80s blues track.",
        metadata={
            "help": (
                "This prompt will be used during evaluation as an additional generated sample."
            )
        },
    )

    # 用于远程文件的 HTTP 承载授权的令牌。如果未指定，将使用运行 `huggingface-cli login` 时生成的令牌（存储在 `~/.huggingface` 中）。
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )

    # `use_auth_token` 参数已弃用，将在 v4.34 中删除。请使用 `token` 代替。
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )

    # 是否允许在 Hub 上定义自己的模型文件中的自定义模型。此选项应仅对您信任的存储库以及您已阅读代码的存储库设置为 `True`。
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )

    # 如果设置并且 `wandb` 在 args.report_to 中，则将生成的音频样本添加到 wandb 日志中。在训练开始和结束时生成音频以显示演变。
    add_audio_samples_to_wandb: bool = field(
        default=False,
        metadata={
            "help": "If set and if `wandb` in args.report_to, will add generated audio samples to wandb logs."
            "Generates audio at the beginning and the end of the training to show evolution."
        },
    )

    # 如果 `True`，则使用 librosa 和 msclap 自动生成歌曲描述。不要忘记安装这些库：`pip install msclap librosa`。
    add_metadata: bool = field(
        default=False,
        metadata={
            "help": (
                "If `True`, automatically generates song descriptions, using librosa and msclap."
                "Don't forget to install these libraries: `pip install msclap librosa`"
            )
        },
    )

    # 如果指定并且 `add_metadata=True`，则将丰富的数据集推送到 Hub。如果只想计算一次，这很有用。
    push_metadata_repo_id: str = field(
        default=None,
        metadata={
            "help": (
                "if specified and `add_metada=True`, will push the enriched dataset to the hub. Useful if you want to compute it only once."
            )
        },
    )

    # 如果使用 `wandb` 进行日志记录，则表示要生成的测试集样本数量。
    num_samples_to_generate: int = field(
        default=4,
        metadata={
            "help": (
                "If logging with `wandb`, indicates the number of samples from the test set to generate"
            )
        },
    )

    # 如果设置，则使用 demucs 执行音频分离。
    audio_separation: bool = field(
        default=False,
        metadata={"help": ("If set, performs audio separation using demucs.")},
    )

    # 如果 `audio_separation`，则表示传递给 demucs 的批量大小。
    audio_separation_batch_size: int = field(
        default=10,
        metadata={
            "help": (
                "If `audio_separation`, indicates the batch size passed to demucs."
            )
        },
    )


@dataclass
class DataCollatorMusicGenWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """
    """
    数据收集器，将动态填充接收到的输入。
    
    Args:
        processor (:class:`~transformers.AutoProcessor`):
            用于处理数据的处理器。
        padding (:obj:`bool`, :obj:`str` 或 :class:`~transformers.tokenization_utils_base.PaddingStrategy`, 可选, 默认为 :obj:`True`):
            选择填充返回序列的策略（根据模型的填充侧和填充索引），选项包括：
            * :obj:`True` 或 :obj:`'longest'`: 填充到批次中最长的序列（如果只提供一个序列，则不填充）。
            * :obj:`'max_length'`: 填充到通过参数 :obj:`max_length` 指定的最大长度，或者如果未提供该参数，则填充到模型可接受的最大输入长度。
            * :obj:`False` 或 :obj:`'do_not_pad'`（默认）: 不填充（即，可以输出包含不同长度序列的批次）。
    """
    # 用于处理数据的处理器
    processor: AutoProcessor
    # 填充策略，默认为 "longest"
    padding: Union[bool, str] = "longest"
    # 特征提取器输入名称，默认为 "input_values"
    feature_extractor_input_name: Optional[str] = "input_values"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        处理输入数据并返回填充后的批次。

        Args:
            features (List[Dict[str, Union[List[int], torch.Tensor]]]): 输入特征列表，每个特征都是一个字典，包含输入ID和标签。

        Returns:
            Dict[str, torch.Tensor]: 填充后的批次数据。
        """
        # 分离输入和标签，因为它们需要不同的长度和不同的填充方法
        labels = [
            torch.tensor(feature["labels"]).transpose(0, 1) for feature in features
        ]
        # (bsz, seq_len, num_codebooks)
        # (batch_size, sequence_length, num_codebooks)
        # 对标签进行填充，填充值为 -100
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        # 对输入ID进行填充
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        input_ids = self.processor.tokenizer.pad(input_ids, return_tensors="pt")

        # 构建批次字典
        batch = {"labels": labels, **input_ids}

        # 如果特征中包含特征提取器的输入名称，则对输入值进行填充
        if self.feature_extractor_input_name in features[0]:
            input_values = [
                {
                    self.feature_extractor_input_name: feature[
                        self.feature_extractor_input_name
                    ]
                }
                for feature in features
            ]
            input_values = self.processor.feature_extractor.pad(
                input_values, return_tensors="pt"
            )

            # 将填充后的输入值添加到批次字典中
            batch[self.feature_extractor_input_name : input_values]

        return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    """
    主函数，负责解析参数、设置日志、检测检查点以及启动训练或评估过程。
    """
    # 查看所有可能的参数，可以在 src/transformers/training_args.py 中找到
    # 或者通过向此脚本传递 --help 标志来查看。
    # 我们现在保持不同的参数集，以便更清晰地分离关注点。

    # 初始化参数解析器，传入模型参数、数据参数和训练参数类
    parser = HfArgumentParser(
        (ModelArguments, DataSeq2SeqTrainingArguments, Seq2SeqTrainingArguments)
    )

    # 判断命令行参数数量和格式
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果我们只向脚本传递一个参数，并且它是一个 JSON 文件的路径，
        # 那么我们解析这个 JSON 文件来获取我们的参数。
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        # 否则，从命令行参数中解析参数到数据类中
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 发送遥测数据。跟踪示例用法有助于我们更好地分配资源以维护它们。
    # 发送的信息包括作为参数传递的信息以及您的 Python/PyTorch 版本。
    send_example_telemetry("run_musicgen_melody", model_args, data_args)

    # Detecting last checkpoint.
    # 检测上一个检查点
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir) # 检查输出目录是否存在
        and training_args.do_train # 检查是否需要训练
        and not training_args.overwrite_output_dir # 检查是否不需要覆盖输出目录
    ):
        # 获取最后一个检查点
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            # 如果没有找到检查点，但输出目录不为空，则抛出错误
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            # 如果找到了检查点，则记录日志信息
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    # 设置日志记录
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", # 日志格式
        datefmt="%m/%d/%Y %H:%M:%S", # 日期格式
        handlers=[logging.StreamHandler(sys.stdout)], # 日志处理器，输出到标准输出
    )
    # 在每个进程上记录简要摘要
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Log on each process the small summary:
    # 在每个进程上记录简要摘要
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    # 仅在主进程上将 Transformers 日志的详细程度设置为信息级别
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    # 记录训练/评估参数
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    # 在初始化模型之前设置随机种子，以确保结果的可重复性
    set_seed(training_args.seed)

    # 1. First, let's load the dataset
    # 1. 首先，让我们加载数据集
    # 创建一个空的 DatasetDict 对象，用于存储原始数据集
    raw_datasets = DatasetDict()
    # 获取数据预处理的进程数
    num_workers = data_args.preprocessing_num_workers
    # 获取是否添加元数据的标志
    add_metadata = data_args.add_metadata

    # 检查 add_metadata 和 text_column_name 是否同时为 True
    if add_metadata and data_args.text_column_name:
        raise ValueError(
            "add_metadata and text_column_name are both True, chose the former if you want automatically generated music descriptions or the latter if you want to use your own set of descriptions."
        )

    if training_args.do_train:
        # 如果需要训练，则加载训练数据集
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,  # 数据集名称
            data_args.dataset_config_name,  # 数据集配置名称
            split=data_args.train_split_name,  # 训练数据集分割名称
            num_proc=num_workers,  # 预处理进程数
        )

        # 检查目标音频列是否存在于训练数据集中
        if data_args.target_audio_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--target_audio_column_name '{data_args.target_audio_column_name}' not found in dataset '{data_args.dataset_name}'."
                " Make sure to set `--target_audio_column_name` to the correct audio column - one of"
                f" {', '.join(raw_datasets['train'].column_names)}."
            )

        # 检查是否提供了实例提示或文本列
        if data_args.instance_prompt is not None:
            logger.warning(
                f"Using the following instance prompt: {data_args.instance_prompt}"
            )
        elif data_args.text_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
                "Make sure to set `--text_column_name` to the correct text column - one of "
                f"{', '.join(raw_datasets['train'].column_names)}."
            )
        elif data_args.text_column_name is None and data_args.instance_prompt is None:
            raise ValueError("--instance_prompt or --text_column_name must be set.")

        # 如果设置了最大训练样本数，则截断数据集
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = (
                raw_datasets["train"]
                .shuffle() # 打乱数据集
                .select(range(data_args.max_train_samples)) # 选择前 max_train_samples 个样本
            )

    if training_args.do_eval:
        # 如果需要评估，则加载评估数据集
        raw_datasets["eval"] = load_dataset(
            data_args.dataset_name,  # 数据集名称
            data_args.dataset_config_name,  # 数据集配置名称
            split=data_args.eval_split_name,  # 评估数据集分割名称
            num_proc=num_workers,  # 预处理进程数
        )

        # 如果设置了最大评估样本数，则截断数据集
        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = raw_datasets["eval"].select(
                # 选择前 max_eval_samples 个样本
                range(data_args.max_eval_samples)
            )

    if data_args.audio_separation:
        try:
            from demucs import pretrained
        except ImportError:
            print(
                "To perform audio separation, you should install additional packages, run: `pip install -e .[metadata]]` or `pip install demucs`."
            )
        # 从 demucs.apply 导入 apply_model 函数，用于应用分离模型
        from demucs.apply import apply_model
        # 从 demucs.audio 导入 convert_audio 函数，用于转换音频格式
        from demucs.audio import convert_audio
        # 从 datasets 导入 Audio 类，用于处理音频数据
        from datasets import Audio

        # 加载预训练的 demucs 模型
        demucs = pretrained.get_model("htdemucs")
        # 检查是否有可用的 GPU
        if torch.cuda.device_count() > 0:
            # 如果有 GPU，将模型移动到第一个 GPU
            demucs.to("cuda:0")

        # 定义目标音频列的名称
        audio_column_name = data_args.target_audio_column_name

        # 定义一个函数，用于将音频数据转换为适合 demucs 处理的格式
        def wrap_audio(audio, sr):
            return {"array": audio.cpu().numpy(), "sampling_rate": sr}

        # 定义一个函数，用于过滤音频的各个部分（人声、鼓、贝斯等）
        def filter_stems(batch, rank=None):
            # 如果有可用的 GPU，则使用 GPU，否则使用 CPU
            device = "cpu" if torch.cuda.device_count() == 0 else "cuda:0"
            if rank is not None:
                # move the model to the right GPU if not there already
                # 如果有多个 GPU，则根据进程编号分配 GPU
                device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
                # move to device and create pipeline here because the pipeline moves to the first GPU it finds anyway
                # 将模型移动到指定的 GPU
                demucs.to(device)

            # 检查音频数据是否为列表类型
            if isinstance(batch[audio_column_name], list):
                # 将音频数据转换为 PyTorch 张量，并移动到指定的设备
                wavs = [
                    convert_audio(
                        torch.tensor(audio["array"][None], device=device).to(
                            torch.float32
                        ),
                        audio["sampling_rate"],  # 原始采样率
                        demucs.samplerate,  # demucs 模型的采样率
                        demucs.audio_channels,  # 音频通道数
                    ).T
                    for audio in batch["audio"]
                ]
                # 获取每个音频的长度
                wavs_length = [audio.shape[0] for audio in wavs]

                # 对音频数据进行填充，使其长度一
                wavs = torch.nn.utils.rnn.pad_sequence(
                    wavs, batch_first=True, padding_value=0.0
                ).transpose(1, 2)
                # 应用 demucs 模型进行音频分离
                stems = apply_model(demucs, wavs)

                # 对分离后的音频进行后处理
                batch[audio_column_name] = [
                    wrap_audio(s[:-1, :, :length].sum(0).mean(0), demucs.samplerate)
                    for (s, length) in zip(stems, wavs_length)
                ]

            else:
                # 如果音频数据不是列表类型，则直接处理单个音频文件
                audio = torch.tensor(
                    batch[audio_column_name]["array"].squeeze(), device=device
                ).to(torch.float32)
                sample_rate = batch[audio_column_name]["sampling_rate"]
                audio = convert_audio(
                    audio, sample_rate, demucs.samplerate, demucs.audio_channels
                )
                # 应用 demucs 模型进行音频分离
                stems = apply_model(demucs, audio[None])

                # 对分离后的音频进行后处理
                batch[audio_column_name] = wrap_audio(
                    stems[0, :-1].mean(0), demucs.samplerate
                )

            return batch

        # 计算进程数，如果有多块 GPU，则使用 GPU 数，否则使用预处理的进程数
        num_proc = (
            torch.cuda.device_count() if torch.cuda.device_count() >= 1 else num_workers
        )

        # 应用音频分离函数到数据集
        raw_datasets = raw_datasets.map(
            filter_stems,
            batched=True,
            batch_size=data_args.audio_separation_batch_size,
            with_rank=True,
            num_proc=num_proc,
        )
        # 将音频列转换为 Audio 类型
        raw_datasets = raw_datasets.cast_column(audio_column_name, Audio())
        
        # 删除模型以释放内存
        del demucs

    if add_metadata:
        try:
            from msclap import CLAP
            import librosa
        except ImportError:
            print(
                "To add metadata, you should install additional packages, run: `pip install -e .[metadata]]"
            )
        # 从 labels 模块导入乐器类别、类型标签和情绪主题类别
        from labels import instrument_classes, genre_labels, mood_theme_classes
        # 导入 tempfile 模块，用于创建临时目录和文件
        import tempfile
        # 导入 torchaudio，用于处理音频文件
        import torchaudio
        # 导入 random，用于随机打乱元数据
        import random

        # 初始化 CLAP 模型，version 为 "2023"，不使用 CUDA
        clap_model = CLAP(version="2023", use_cuda=False)
        # 获取乐器类别的文本嵌入
        instrument_embeddings = clap_model.get_text_embeddings(instrument_classes)
        # 获取类型标签的文本嵌入
        genre_embeddings = clap_model.get_text_embeddings(genre_labels)
        # 获取情绪主题类别的文本嵌入
        mood_embeddings = clap_model.get_text_embeddings(mood_theme_classes)

        # 定义一个函数，用于丰富文本数据
        def enrich_text(batch):
            # 从 batch 中提取音频数据和采样率
            audio, sampling_rate = (
                batch["audio"]["array"],
                batch["audio"]["sampling_rate"],
            )

            # 使用 librosa 计算节奏（BPM）和和弦特征
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sampling_rate)
            tempo = f"{str(round(tempo))} bpm"  # 通常不准确
            chroma = librosa.feature.chroma_stft(y=audio, sr=sampling_rate)
            # 计算最可能的调性
            key = np.argmax(np.sum(chroma, axis=1))
            key = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][key]

            # 使用临时目录保存音频文件
            with tempfile.TemporaryDirectory() as tempdir:
                path = os.path.join(tempdir, "tmp.wav")
                torchaudio.save(path, torch.tensor(audio).unsqueeze(0), sampling_rate)
                # 获取音频嵌入
                audio_embeddings = clap_model.get_audio_embeddings([path])

            # 计算与乐器类别的相似度，并选择最相似的类别
            instrument = clap_model.compute_similarity(
                audio_embeddings, instrument_embeddings
            ).argmax(dim=1)[0]

            # 计算与类型标签的相似度，并选择最相似的类别
            genre = clap_model.compute_similarity(
                audio_embeddings, genre_embeddings
            ).argmax(dim=1)[0]

            # 计算与情绪主题类别的相似度，并选择最相似的类别
            mood = clap_model.compute_similarity(
                audio_embeddings, mood_embeddings
            ).argmax(dim=1)[0]

            # 将索引转换为实际的类别名称
            instrument = instrument_classes[instrument]
            genre = genre_labels[genre]
            mood = mood_theme_classes[mood]

            # 组合元数据
            metadata = [key, tempo, instrument, genre, mood]

            # 随机打乱元数据顺序
            random.shuffle(metadata)
            # 将元数据列表转换为字符串
            batch["metadata"] = ", ".join(metadata)
            return batch

        # 使用 enrich_text 函数处理数据集
        raw_datasets = raw_datasets.map(
            enrich_text,
            # 如果有 GPU，则使用单进程，否则使用预处理进程数
            num_proc=1 if torch.cuda.device_count() > 0 else num_workers,
            desc="add metadata",
        )

        # 删除模型和嵌入以释放内存
        del clap_model, instrument_embeddings, genre_embeddings, mood_embeddings

        # 如果指定了 push_metadata_repo_id，则将丰富后的数据集推送到 Hub
        if data_args.push_metadata_repo_id:
            raw_datasets.push_to_hub(data_args.push_metadata_repo_id)

    # 3. Next, let's load the config as we might need it to create
    # load config
    # 3. 接下来，让我们加载配置，因为我们可能需要它来创建模型
    # 加载配置
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, # 模型名称或路径
        cache_dir=model_args.cache_dir, # 缓存目录
        token=data_args.token, 
        trust_remote_code=data_args.trust_remote_code, # 是否信任远程代码
        revision=model_args.model_revision, # 模型版本
    )

    # update pad token id and decoder_start_token_id
    # 更新填充标记ID和解码器起始标记ID
    config.update(
        {
            "pad_token_id": model_args.pad_token_id
            if model_args.pad_token_id is not None
            else model.config.pad_token_id, # 如果指定了填充标记ID，则使用它，否则使用模型的填充标记ID
            "decoder_start_token_id": model_args.decoder_start_token_id
            if model_args.decoder_start_token_id is not None
            else model.config.decoder_start_token_id, # 如果指定了解码器起始标记ID，则使用它，否则使用模型解码器的起始标记ID
        }
    )
    config.decoder.update(
        {
            "pad_token_id": model_args.pad_token_id
            if model_args.pad_token_id is not None
            else model.config.decoder.pad_token_id, # 如果指定了填充标记ID，则使用它，否则使用模型解码器的填充标记ID
            "decoder_start_token_id": model_args.decoder_start_token_id
            if model_args.decoder_start_token_id is not None
            else model.config.decoder.decoder_start_token_id, # 如果指定了解码器起始标记ID，则使用它，否则使用模型解码器的解码器起始标记ID
        }
    )

    # 4. Now we can instantiate the processor and model
    # 4. 现在我们可以实例化处理器和模型

    # Note for distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently download model & vocab.
    # 注意对于分布式训练，`.from_pretrained` 方法保证只有一个本地进程可以同时下载模型和词汇表。

    # load processor
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        model_args.processor_name or model_args.model_name_or_path, # 使用处理器名称或模型名称/路径
        cache_dir=model_args.cache_dir, # 缓存目录
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code, # 是否信任远程代码
    )

    # 处理实例提示和完整生成样本文本
    instance_prompt = data_args.instance_prompt
    instance_prompt_tokenized = None
    full_generation_sample_text = data_args.full_generation_sample_text
    if data_args.instance_prompt is not None:
        # 如果提供了实例提示，则对其进行分词
        instance_prompt_tokenized = processor.tokenizer(instance_prompt)
    if full_generation_sample_text is not None:
        # 如果提供了完整生成样本文本，则对其进行分词并返回张量
        full_generation_sample_text = processor.tokenizer(
            full_generation_sample_text, return_tensors="pt"
        )

    # create model
    # 创建模型
    model = AutoModelForTextToWaveform.from_pretrained(
        model_args.model_name_or_path, # 模型名称或路径
        cache_dir=model_args.cache_dir, # 缓存目录
        config=config, # 模型配置
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code, # 是否信任远程代码
        revision=model_args.model_revision, # 模型版本
    )

    # take audio_encoder_feature_extractor
    # 获取音频编码器的特征提取器
    audio_encoder_feature_extractor = AutoFeatureExtractor.from_pretrained(
        model.config.audio_encoder._name_or_path, # 音频编码器的名称或路径
    )

    # 5. Now we preprocess the datasets including loading the audio, resampling and normalization
    # 5. 现在我们预处理数据集，包括加载音频、重采样和归一化

    # Thankfully, `datasets` takes care of automatically loading and resampling the audio,
    # so that we just need to set the correct target sampling rate and normalize the input
    # via the `feature_extractor`
    # 值得庆幸的是，`datasets` 自动处理加载和重采样音频，
    # 所以我们只需要通过 `feature_extractor` 设置正确的目标采样率和归一化输入

    # resample target audio
    # 重采样目标音频
    dataset_sampling_rate = (
        next(iter(raw_datasets.values())) # 获取第一个数据集
        .features[data_args.target_audio_column_name] # 获取目标音频列的特征
        .sampling_rate # 获取采样率
    )
    if dataset_sampling_rate != audio_encoder_feature_extractor.sampling_rate:
        # 如果数据集的采样率与特征提取器的采样率不同，则进行重采样
        raw_datasets = raw_datasets.cast_column(
            data_args.target_audio_column_name,
            datasets.features.Audio(
                sampling_rate=audio_encoder_feature_extractor.sampling_rate
            ),
        )

    # 如果有条件音频列，则也进行重采样
    if data_args.conditional_audio_column_name is not None:
        dataset_sampling_rate = (
            next(iter(raw_datasets.values())) # 获取第一个数据集
            .features[data_args.conditional_audio_column_name] # 获取条件音频列的特征
            .sampling_rate # 获取采样率
        )
        if dataset_sampling_rate != processor.feature_extractor.sampling_rate:
            # 如果数据集的采样率与处理器的特征提取器的采样率不同，则进行重采样
            raw_datasets = raw_datasets.cast_column(
                data_args.conditional_audio_column_name,
                datasets.features.Audio(
                    sampling_rate=processor.feature_extractor.sampling_rate
                ),
            )

    # derive max & min input length for sample rate & max duration
    # 根据采样率和最大时长计算输入的最大和最小长度
    max_target_length = (
        data_args.max_duration_in_seconds # 最大时长（秒）
        * audio_encoder_feature_extractor.sampling_rate # 特征提取器的采样率
    )
    min_target_length = (
        data_args.min_duration_in_seconds # 最小时长（秒）
        * audio_encoder_feature_extractor.sampling_rate # 特征提取器的采样率
    )
    # 目标音频列的名称
    target_audio_column_name = data_args.target_audio_column_name
    # 条件音频列的名称
    conditional_audio_column_name = data_args.conditional_audio_column_name
    # 文本列的名称
    text_column_name = data_args.text_column_name
    # 特征提取器的输入名称
    feature_extractor_input_name = processor.feature_extractor.model_input_names[0]
    # 解码器的填充标记ID
    audio_encoder_pad_token_id = config.decoder.pad_token_id
    # 解码器的码本数量
    num_codebooks = model.decoder.config.num_codebooks

    # 如果提供了实例提示，则在主进程中预处理实例提示
    if data_args.instance_prompt is not None:
        with training_args.main_process_first(desc="instance_prompt preprocessing"):
            # compute text embeddings on one process since it's only a forward pass
            # do it on CPU for simplicity
            # 在一个进程中计算文本嵌入，因为只需要前向传播
            # 为了简化，在CPU上进行
            instance_prompt_tokenized = instance_prompt_tokenized["input_ids"]

    # Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    # 预处理数据集。
    # 我们需要将音频文件读取为数组，并对目标进行分词。
    def prepare_audio_features(batch):
        # 加载音频
        if conditional_audio_column_name is not None:
            sample = batch[conditional_audio_column_name]  # 获取条件音频样本
            inputs = processor.feature_extractor(
                sample["array"], sampling_rate=sample["sampling_rate"]  # 音频数组和采样率
            )
            batch[feature_extractor_input_name] = getattr(
                inputs, feature_extractor_input_name # 获取特征提取器的输入名称
            )[0]

        if text_column_name is not None:
            # 获取文本数据
            text = batch[text_column_name]
            # 对文本进行分词
            batch["input_ids"] = processor.tokenizer(text)["input_ids"]
        elif add_metadata is not None and "metadata" in batch:
            # 获取元数据
            metadata = batch["metadata"]
            if instance_prompt is not None and instance_prompt != "":
                # 如果有实例提示，则将其添加到元数据中
                metadata = f"{instance_prompt}, {metadata}"
            # 对元数据进行分词
            batch["input_ids"] = processor.tokenizer(metadata)["input_ids"]
        else:
            # 如果没有文本或元数据，则使用实例提示的分词
            batch["input_ids"] = instance_prompt_tokenized

        # load audio
        # 加载目标音频
        target_sample = batch[target_audio_column_name]
        # 音频数组和采样率
        labels = audio_encoder_feature_extractor(
            target_sample["array"], sampling_rate=target_sample["sampling_rate"]
        )
        # 设置标签
        batch["labels"] = labels["input_values"]

        # take length of raw audio waveform
        # 获取原始音频波形的长度
        batch["target_length"] = len(target_sample["array"].squeeze())
        return batch

    # 在主进程中预处理数据集
    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_audio_features,
            remove_columns=next(iter(raw_datasets.values())).column_names, # 移除原始列
            num_proc=num_workers, # 使用预处理进程数
            desc="preprocess datasets",
        )

        # 定义一个函数，用于检查音频长度是否在范围内
        def is_audio_in_length_range(length):
            return length > min_target_length and length < max_target_length

        # filter data that is shorter than min_target_length
        # 过滤掉长度小于最小目标长度的数据
        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["target_length"], # 使用目标长度列进行过滤
        )

    # 加载音频解码器
    audio_decoder = model.audio_encoder

    # 创建填充标签张量，形状为 (1, 1, num_codebooks, 1)，填充值为 audio_encoder_pad_token_id
    pad_labels = torch.ones((1, 1, num_codebooks, 1)) * audio_encoder_pad_token_id
    
    # 如果只有一个GPU，则将音频解码器移动到GPU
    if torch.cuda.device_count() == 1:
        audio_decoder.to("cuda")

    # 定义一个函数，用于在批处理数据上应用音频解码器
    def apply_audio_decoder(batch, rank=None):
        """
        在批处理数据上应用音频解码器。

        Args:
            batch (dict): 包含 'labels' 键的批处理数据字典。
            rank (int, optional): 当前进程的排名，用于多GPU训练。

        Returns:
            dict: 更新后的批处理数据字典，包含处理后的 'labels'。
        """
        if rank is not None:
            # move the model to the right GPU if not there already
            # 如果有多个GPU，则根据进程编号分配GPU
            device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
            # 将音频解码器移动到指定的GPU
            audio_decoder.to(device)

        with torch.no_grad():
            # 对标签进行编码
            labels = audio_decoder.encode(
                torch.tensor(batch["labels"]).to(audio_decoder.device)
            )["audio_codes"]

        # add pad token column
        # 添加填充标记列
        labels = torch.cat(
            [pad_labels.to(labels.device).to(labels.dtype), labels], dim=-1
        )

        # 构建延迟模式掩码
        labels, delay_pattern_mask = model.decoder.build_delay_pattern_mask(
            labels.squeeze(0),
            audio_encoder_pad_token_id,
            labels.shape[-1] + num_codebooks,
        )

        # 应用延迟模式掩码
        labels = model.decoder.apply_delay_pattern_mask(labels, delay_pattern_mask)

        # the first timestamp is associated to a row full of BOS, let's get rid of it
        # 删除第一个时间步的BOS标记
        batch["labels"] = labels[:, 1:].cpu()
        return batch

    # 在主进程中预处理音频目标
    with training_args.main_process_first(desc="audio target preprocessing"):
        # Encodec doesn't truely support batching
        # Pass samples one by one to the GPU
        # Encodec 不真正支持批处理
        # 将样本逐个传递到GPU
        vectorized_datasets = vectorized_datasets.map(
            apply_audio_decoder,
            with_rank=True,
            num_proc=torch.cuda.device_count() # 如果有GPU，则使用GPU数量
            if torch.cuda.device_count() > 0
            else num_workers, # 否则使用预处理进程数
            desc="Apply encodec",
        )

    # 如果需要将音频样本添加到W&B日志中，并且W&B在报告中
    if data_args.add_audio_samples_to_wandb and "wandb" in training_args.report_to:
        if is_wandb_available():
            from transformers.integrations import WandbCallback 
        else:
            raise ValueError(
                "`args.add_audio_samples_to_wandb=True` and `wandb` in `report_to` but wandb is not installed. See https://docs.wandb.ai/quickstart to install."
            )

    # 6. Next, we can prepare the training.
    # 接下来，我们可以准备训练。

    # Let's use word CLAP similary as our evaluation metric,
    # instantiate a data collator and the trainer
    # 使用CLAP相似度作为评估指标，
    # 实例化数据收集器和训练器


    # Define evaluation metrics during training, *i.e.* CLAP similarity
    # 定义训练期间的评估指标，即CLAP相似度
    clap = AutoModel.from_pretrained(model_args.clap_model_name_or_path)
    clap_processor = AutoProcessor.from_pretrained(model_args.clap_model_name_or_path)

    def clap_similarity(texts, audios):
        """
        计算文本和音频之间的CLAP相似度。

        Args:
            texts (list): 文本列表。
            audios (torch.Tensor): 音频张量。

        Returns:
            float: 平均CLAP相似度。
        """
        clap_inputs = clap_processor(
            text=texts, audios=audios.squeeze(1), padding=True, return_tensors="pt"
        )
        text_features = clap.get_text_features(
            clap_inputs["input_ids"],
            attention_mask=clap_inputs.get("attention_mask", None),
        )
        audio_features = clap.get_audio_features(clap_inputs["input_features"])

        cosine_sim = torch.nn.functional.cosine_similarity(
            audio_features, text_features, dim=1, eps=1e-8
        )

        return cosine_sim.mean()

    eval_metrics = {"clap": clap_similarity}

    def compute_metrics(pred):
        """
        计算评估指标。

        Args:
            pred: 预测结果。

        Returns:
            dict: 包含评估指标结果的字典。
        """
        input_ids = pred.inputs
        # 替换填充标记
        input_ids[input_ids == -100] = processor.tokenizer.pad_token_id
        # 解码文本
        texts = processor.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # 获取音频预测
        audios = pred.predictions

        # 计算指标
        results = {key: metric(texts, audios) for (key, metric) in eval_metrics.items()}

        return results

    # Now save everything to be able to create a single processor later
    # make sure all processes wait until data is saved
    # 现在保存所有内容，以便能够创建单个处理器
    # 确保所有进程等待数据保存完毕
    with training_args.main_process_first():
        # only the main process saves them
        # 只有主进程保存它们
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            # 保存特征提取器、分词器和配置
            processor.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    # Instantiate custom data collator
    # 实例化自定义数据收集器
    data_collator = DataCollatorMusicGenWithPadding(
        processor=processor, # 处理器
        feature_extractor_input_name=feature_extractor_input_name, # 特征提取器输入名称
    ) 

    # Freeze Encoders
    # 冻结编码器
    model.freeze_audio_encoder() # 冻结音频编码器
    if model_args.freeze_text_encoder:
        model.freeze_text_encoder() # 如果需要，冻结文本编码器

    # 设置指导尺度（如果有指定）
    if model_args.guidance_scale is not None:
        model.generation_config.guidance_scale = model_args.guidance_scale
    
    # 如果使用Lora（低秩适应），则配置Lora
    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model

        # TODO(YL): add modularity here
        # TODO(YL): 在这里添加模块化
        target_modules = (
            [
                "enc_to_dec_proj", # 编码器到解码器投影层
                "audio_enc_to_dec_proj", # 音频编码器到解码器投影层
                "k_proj", # K投影层
                "v_proj", # V投影层
                "q_proj", # Q投影层
                "out_proj", # 输出投影层
                "fc1", # 全连接层1
                "fc2", # 全连接层2
                "lm_heads.0", # 语言模型头0
            ]
            + [f"lm_heads.{str(i)}" for i in range(len(model.decoder.lm_heads))] # 所有语言模型头
            + [f"embed_tokens.{str(i)}" for i in range(len(model.decoder.lm_heads))] # 所有嵌入词元
        )

        if not model_args.freeze_text_encoder:
            # 如果不冻结文本编码器，则添加更多目标模块
            target_modules.extend(["k", "v", "q", "o", "wi", "wo"])

        config = LoraConfig(
            r=16, # Lora秩
            lora_alpha=16, # Lora alpha
            target_modules=target_modules, # 目标模块列表
            lora_dropout=0.05, # Lora dropout率
            bias="none", # 偏置
        )
        # 启用输入梯度要求
        model.enable_input_require_grads() 
        # 应用Lora配置到模型
        model = get_peft_model(model, config)
        # 打印可训练的参数
        model.print_trainable_parameters()
        # 记录使用Lora的模块
        logger.info(f"Modules with Lora: {model.targeted_module_names}")

    # Initialize MusicgenTrainer
    # 初始化MusicgenTrainer
    trainer = MusicgenTrainer(
        model=model, # 模型
        data_collator=data_collator, # 数据收集器
        args=training_args, # 训练参数
        compute_metrics=compute_metrics, # 计算指标函数
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None, # 训练数据集
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None, # 评估数据集
        tokenizer=processor, # 分词器
    )

    # 如果需要将音频样本添加到W&B日志中，并且W&B在报告中
    if data_args.add_audio_samples_to_wandb and "wandb" in training_args.report_to:

        def decode_predictions(processor, predictions):
            audios = predictions.predictions
            # 解码预测结果，提取音频数据
            return {"audio": np.array(audios.squeeze(1))}

        class WandbPredictionProgressCallback(WandbCallback):
            """
            Custom WandbCallback to log model predictions during training.
            """
            """
            自定义W&B回调，用于在训练过程中记录模型预测结果。
            """

            def __init__(
                self,
                trainer,
                processor,
                val_dataset,
                additional_generation,
                max_new_tokens,
                num_samples=8,
            ):
                """Initializes the WandbPredictionProgressCallback instance.

                Args:
                    trainer (Seq2SeqTrainer): The Hugging Face Seq2SeqTrainer instance.
                    processor (AutoProcessor): The processor associated
                    with the model.
                    val_dataset (Dataset): The validation dataset.
                    num_samples (int, optional): Number of samples to select from
                    the validation dataset for generating predictions.
                    Defaults to 8.
                """
                """
                初始化WandbPredictionProgressCallback实例。

                Args:
                    trainer (Seq2SeqTrainer): Hugging Face的Seq2SeqTrainer实例。
                    processor (AutoProcessor): 与模型关联的处理器。
                    val_dataset (Dataset): 验证数据集。
                    num_samples (int, optional): 从验证数据集中选择的样本数量，用于生成预测。默认为8。
                """
                super().__init__()
                self.trainer = trainer
                self.processor = processor
                self.additional_generation = additional_generation
                # 从验证数据集中选择样本
                self.sample_dataset = val_dataset.select(range(num_samples))
                self.max_new_tokens = max_new_tokens

            def on_train_begin(self, args, state, control, **kwargs):
                """
                在训练开始时调用的回调方法。

                Args:
                    args: 训练参数。
                    state: 训练状态。
                    control: 训练控制。
                    **kwargs: 其他关键字参数。
                """
                super().on_train_begin(args, state, control, **kwargs)
                # 设置随机种子
                set_seed(training_args.seed)
                # 对样本进行预测
                predictions = self.trainer.predict(self.sample_dataset)
                # decode predictions and labels
                # 解码预测结果和标签
                predictions = decode_predictions(self.processor, predictions)

                input_ids = self.sample_dataset["input_ids"]
                texts = self.processor.tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True # 解码文本，忽略特殊标记
                )
                # 提取音频数据
                audios = [a for a in predictions["audio"]]

                additional_audio = self.trainer.model.generate(
                    **self.additional_generation.to(self.trainer.model.device),
                    max_new_tokens=self.max_new_tokens, # 设置生成的最大新标记数
                )
                additional_text = self.processor.tokenizer.decode(
                    self.additional_generation["input_ids"].squeeze(), # 解码额外样本的文本
                    skip_special_tokens=True,
                )
                # 添加额外文本
                texts.append(additional_text)
                # 添加额外音频
                audios.append(additional_audio.squeeze().cpu())

                # log the table to wandb
                # 将音频和文本记录到W&B
                self._wandb.log(
                    {
                        "sample_songs": [
                            self._wandb.Audio(
                                audio, # 音频数据
                                caption=text, # 文本描述
                                sample_rate=audio_encoder_feature_extractor.sampling_rate, # 采样率
                            )
                            for (audio, text) in zip(audios, texts)
                        ]
                    }
                )

            def on_train_end(self, args, state, control, **kwargs):
                """
                在训练结束时调用的回调方法。

                Args:
                    args: 训练参数。
                    state: 训练状态。
                    control: 训练控制。
                    **kwargs: 其他关键字参数。
                """
                super().on_train_end(args, state, control, **kwargs)
                # 设置随机种子
                set_seed(training_args.seed)
                # 对样本进行预测
                predictions = self.trainer.predict(self.sample_dataset)
                # decode predictions and labels
                # 解码预测结果和标签
                predictions = decode_predictions(self.processor, predictions)

                input_ids = self.sample_dataset["input_ids"]
                texts = self.processor.tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True # 解码文本，忽略特殊标记
                )
                # 提取音频数据
                audios = [a for a in predictions["audio"]]

                additional_audio = self.trainer.model.generate(
                    **self.additional_generation.to(self.trainer.model.device),
                    max_new_tokens=self.max_new_tokens, # 设置生成的最大新标记数
                )
                additional_text = self.processor.tokenizer.decode(
                    self.additional_generation["input_ids"].squeeze(), # 解码额外样本的文本
                    skip_special_tokens=True,
                )
                # 添加额外文本
                texts.append(additional_text)
                # 添加额外音频
                audios.append(additional_audio.squeeze().cpu())

                # log the table to wandb
                # 将音频和文本记录到W&B
                self._wandb.log(
                    {
                        "sample_songs": [
                            self._wandb.Audio(
                                audio, # 音频数据
                                caption=text, # 文本描述
                                sample_rate=audio_encoder_feature_extractor.sampling_rate,
                            )
                            for (audio, text) in zip(audios, texts)
                        ]
                    }
                )

        # Instantiate the WandbPredictionProgressCallback
        # 实例化 WandbPredictionProgressCallback 回调
        progress_callback = WandbPredictionProgressCallback(
            trainer=trainer, # 训练器实例
            processor=processor, # 处理器实例
            val_dataset=vectorized_datasets["eval"], # 验证数据集
            additional_generation=full_generation_sample_text, # 用于生成额外样本的输入
            num_samples=data_args.num_samples_to_generate, # 从验证数据集中选择的样本数量
            max_new_tokens=training_args.generation_max_length, # 生成的最大新标记数
        )

        # Add the callback to the trainer
        # 将回调添加到训练器
        trainer.add_callback(progress_callback)

    # 8. Finally, we can start training
    # 最后，我们可以开始训练

    # Training
    # 训练
    if training_args.do_train:
        # use last checkpoint if exist
        # 如果存在最后的检查点，则使用它
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path): # 如果模型名称或路径是一个目录
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        # 开始训练
        train_result = trainer.train(
            resume_from_checkpoint=checkpoint, # 从检查点恢复训练
            ignore_keys_for_eval=["past_key_values", "attentions"], # 在评估时忽略的键
        )
        # 保存模型
        trainer.save_model()

        # 获取训练指标
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"]) # 如果未指定最大训练样本数，则使用训练数据集的长度
        )
        metrics["train_samples"] = min(
            max_train_samples, len(vectorized_datasets["train"]) # 记录实际训练样本数
        )

        # 记录并保存训练指标
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    # 评估
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # 进行评估
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(vectorized_datasets["eval"]) # 如果未指定最大评估样本数，则使用评估数据集的长度
        )
        metrics["eval_samples"] = min(
            max_eval_samples, len(vectorized_datasets["eval"]) # 记录实际评估样本数
        )   

        # 记录并保存评估指标
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    # 编写模型卡片并（可选）推送到Hub
    config_name = (
        data_args.dataset_config_name
        if data_args.dataset_config_name is not None
        else "na" # 如果未指定数据集配置名称，则使用 "na"
    )
    kwargs = {
        "finetuned_from": model_args.model_name_or_path, # 微调的模型名称或路径
        "tasks": "text-to-music", # 任务类型
        "tags": ["text-to-music", data_args.dataset_name], # 标签
        "dataset_args": (
            f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split:" # 数据集参数
            f" {data_args.eval_split_name}"
        ),
        "dataset": f"{data_args.dataset_name.upper()} - {config_name.upper()}", # 数据集名称
    }

    # 如果需要推送到Hub，则推送；否则，创建模型卡片
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
  
    set_start_method("spawn")
  
    main()
