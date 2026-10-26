● v2-fast 不是独立模型——按报告说明它和 v2 同模型同权重（infer_fast_strat.py:91 直接加载 runs_strat/v2_strat/best），只是推理配置不同（bs16 + FP16
  + stride448 + 无 TTA）。所以它对应的就是 SegFormer 这篇论文：
     
  论文信息

  - 全称：SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers  - 作者：Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo（HKU / NJU / NVIDIA / Caltech）
  - 会议：NeurIPS 2021（第 35 届 Conference on Neural Information Processing Systems）
  - arXiv：https://arxiv.org/abs/2105.15203 （ID: 2105.15203，v3 于 2021-10-28）DOI: 10.48550/arXiv.2105.15203
  - 官方代码：https://github.com/NVlabs/SegFormer
  BibTeX
  bibtex
  @inproceedings{xie2021segformer,
    title     = {SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
    author    = {Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M. and Luo, Ping},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    volume    = {34},
    year      = {2021}
  }

  与当前工程的具体对应关系

  ┌────────────┬─────────────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────┐
  │     项     │                                 值                                  │                          依据                          │
  ├────────────┼─────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ 架构       │ SegFormer + MiT-B1 编码器                                           │ config.json: depths=[2,2,2,2],                         │
  │            │                                                                     │ hidden_sizes=[64,128,320,512] → B1                     │
  ├────────────┼─────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ 实现       │ HuggingFace SegformerForSemanticSegmentation                        │ train_strat.py:13, 91                                  │
  ├────────────┼─────────────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────┤
  │ 初始化权重 │ 本地 segformer_baseline/models/segformer-b1-ade/（ADE20K 微调的 B1  │ config_strat.yaml:31 model_ckpt                        │
  │            │ 检查点）                                                            │                                                        │
  └────────────┴─────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────┘

  两点需要说明：

  1. 本地 config.json 里 _name_or_path 是 None（是拷贝出来的本地目录），所以"来源是 nvidia/segformer-b1-finetuned-ade-512-512"是我根据目录名
  segformer-b1-ade + 已核实的 B1 结构推断的，不是从文件里读到的。引用时如果要写具体的预训练权重出处，建议核对该检查点最初的下载来源。
  2. 论文只涵盖网络结构（MiT 编码器 + 全 MLP 解码器）。本项目 v2 相对 v1 的改进——Focal+Dice+Lovasz 损失、640
  分辨率、copy-paste、TTA——都是本工程自己的训练策略，不来自该论文，写报告时不要把配方归到 SegFormer 上。

  Sources: arXiv:2105.15203 (https://arxiv.org/abs/2105.15203) · NeurIPS 2021 proceedings
  (https://proceedings.neurips.com.cn/paper_files/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html) · HF paper page
  (https://huggingface.co/papers/2105.15203)