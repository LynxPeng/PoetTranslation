# 唐诗宋词英译实验：代码、配置与完成数据

本目录保存本次实验最终使用的翻译生成、DeepSeek评分代码及对应配置、离线测试与原始完成数据。运行环境为Python 3.11，生产入口和所收录测试仅使用标准库。

## 环境与密钥

在仓库根目录运行，以下示例使用PowerShell：

```powershell
conda env create -p .\.conda -f environment.yml
conda activate .\.conda
$env:MIMO_API_KEY="你的MiMo密钥"
$env:DASHSCOPE_API_KEY="你的千问密钥"
$env:DEEPSEEK_API_KEY="你的DeepSeek密钥"
```

密钥仅从环境变量读取。MiMo联网搜索还需要在平台开通联网插件。配置、模型名称和接口地址保留实验执行时版本；数据收集于2026年11月。

## 运行入口

```powershell
# 离线测试，不调用模型
python -X utf8 -m unittest discover -s tests -v

# 普通六组生成：每轨54个去重条件，每首5次，普通任务关闭思考
python -X utf8 generate.py --model mimo --workers 48
python -X utf8 generate.py --model qwen --workers 48
python -X utf8 generate.py --config run_qwen38_all.json --model qwen38 --workers 256

# 对应评分；先完成生成，再运行统一裁判，裁判关闭思考
python -X utf8 evaluate.py --workers 128
python -X utf8 evaluate.py --config run_qwen38_all.json --workers 128

# F12：使用匹配对照，MiMo仅开关；Qwen有4096/16384预算
python -X utf8 generate.py --config run_f12_mimo.json --model mimo --workers 48
python -X utf8 generate.py --config run_f12_qwen.json --model qwen --workers 48
python -X utf8 generate.py --config run_f12_qwen38.json --model qwen38 --workers 48
python -X utf8 evaluate.py --config run_f12_qwen38.json --workers 128

# 最终组合：译前按需搜索/分析→两模型各2候选→MiMo综合→一轮TEaR→评分
python -X utf8 run_combination.py --config run_combination.json --evaluate
```

`configs/run_g1.json`至`run_g6.json`可单独运行各组。最终组合每首仅1次，共16份；MiMo开启思考，Qwen3.8思考预算4096；本轮冻结搜索规则为整批最多16次、每首最多1次，实际14次，仅译前阶段联网。

仓库已经包含完成结果。原配置续跑会跳过已记录任务，不会重新生成整批数据。开展新一轮实验时，应先另存配置，将生成输出、评分输出和步骤台账改为新的路径，保留归档数据。失败项仅在明确需要时用`--retry-failed`补跑。请勿同时启动多个进程写同一输出或台账。

`run_qwen38_all.py`是当时全量运行的编排入口；`f12_supplement.py`、`f12_isolation.py`和`prepare_f12_qwen38.py`保留F12配置构建及补测辅助逻辑。其历史汇总命令可能在`docs/`产生新的运行说明；本仓库不收录报告或图表。

## 数据位置与读取口径

| 文件 | 内容 |
|---|---|
| `results/translations.json_mode_no_thinking.jsonl` | 原MiMo、Qwen六组及F12译文，共9440个生成ID |
| `results/scores.jsonl` | 对应评分及历史失败/修复记录 |
| `results/translations.qwen38_all.jsonl` | Qwen3.8全量4800份，含4320份六组及复用的480份F12 |
| `results/scores.qwen38_all.jsonl` | Qwen3.8对应评分 |
| `results/translations.f12_qwen38.jsonl`、`scores.f12_qwen38.jsonl` | 独立F12源文件；已包含在Qwen3.8全量文件中，不应再次相加 |
| `results/realtime_steps.*.jsonl` | 对应运行阶段的步骤台账，保留原始请求、返回、缓存用量及思考内容 |
| `results/combination_v2/` | 最终组合的16份译文、评分、四候选与搜索/分析台账、分数CSV及原始数据校验值 |

六组基础实验共12960份，独立F12共1280份，合计14240份；最终组合另有16份。跨组别名和独立F12源文件均不能重复计数。

JSONL保留历史行。按同一`id`最后一条取当前状态，评分还需筛选`evaluation/rubric.json`中的当前`protocol_version`，再通过`generation_id`连接译文。不要把历史失败行累计成当前失败数。评分含D1—D7与`total`，七项上限依次为30/10/15/15/10/10/10，总分100；这些是统一量表分数。

`data/benchmark/poems.json`为16首冻结原作；`data/experiment/`保存生成端示例与材料。`gold_annotations.json`和`manifest.json`是冻结设计资料，当前终评不将逐诗标注发送给裁判。当前评分只提供统一规则、原诗和译文；相同原诗、译文及评分配置复用评分。

生成失败保留`status=faild`及原始返回，其他任务继续。单轮和多轮最终译文采用JSON格式；普通中间步骤按自然对话传递，G1-C11反馈为专用JSON解析例外。原始数据原样归档，不删除低分或历史失败。
