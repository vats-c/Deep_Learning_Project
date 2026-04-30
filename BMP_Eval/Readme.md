# BMP_Eval

This folder contains scripts to run **KGI** and **Aspect-KGI** on the **HotpotQA** and **MuSiQue** datasets using local LLMs via Ollama.

---

## 🚀 Quick Start

```bash
cd BMP
python3 -m venv venv
source venv/bin/activate
pip install numpy networkx joblib scikit-learn sentence-transformers tqdm langchain-community langchain-ollama


ollama serve
ollama pull qwen2.5:14b

Folder Structure :-

BMP/
└── BMP_Eval/
    ├── KGI.py
    ├── Aspect_KGI.py
    ├── Hotpot/
    │   ├── triplets_14b.json
    │   └── hotpot.md        # NOT included (this you need to download, which is publicaly available, we couldnt add due to size constrait)
    └── Musique/
        ├── triplets_14b.json
        └── musique.md       # NOT included (this you need to download, which is publicaly available, we couldnt add due to size constrait)


These are publicly available datasets. You must:

Download them from their official sources
Place them in:
BMP_Eval/Hotpot/hotpot.md
BMP_Eval/Musique/musique.md

Expected Format

Each file should contain:

**Question:** ...
**Answer:** ...


This project uses Ollama for LLM inference.

Recommended model:

qwen2.5:14b

Check available models:

ollama list



Running KGI :-

Hotpot
cd BMP_Eval

python3 KGI.py \
  --triplets-json Hotpot/triplets_14b.json \
  --qa-file Hotpot/hotpot.md \
  --output-file Hotpot/output.md \
  --llm-model qwen2.5:14b

MuSiQue
python3 KGI.py \
  --triplets-json Musique/triplets_14b.json \
  --qa-file Musique/musique.md \
  --output-file Musique/output.md \
  --llm-model qwen2.5:14b


Running Aspect-KGI :-
  
Hotpot
python3 Aspect_KGI.py \
  --triplets-json Hotpot/triplets_14b.json \
  --qa-file Hotpot/hotpot.md \
  --output-file Hotpot/output_aspect.md \
  --llm-model qwen2.5:14b
MuSiQue
python3 Aspect_KGI.py \
  --triplets-json Musique/triplets_14b.json \
  --qa-file Musique/musique.md \
  --output-file Musique/output_aspect.md \
  --llm-model qwen2.5:14b


Output :-

Results are saved as markdown files:

Hotpot/output.md
Hotpot/output_aspect.md
Musique/output.md
Musique/output_aspect.md

Each output contains:

Question
Ground Truth
Retrieved Answer


Evaluation :-

Using judge script:

python3 llm-judge.py \
  --input-file BMP_Eval/Hotpot/output.md \
  --output-file BMP_Eval/Hotpot/eval.json \
  --judge-model qwen2.5:14b

For Aspect-KGI, use output_aspect.md

        
