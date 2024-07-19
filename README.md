# CoD, Towards an Interpretable Medical Agent using Chain of Diagnosis

<div align="center">
<h5>
  📃 <a href="https://arxiv.org/abs/2407.13301" target="_blank">Paper</a>  • 🖥️ <a href="#" target="_blank">Demo(Coming)</a>
</h5>
</div>

<div align="center">
<h4>
  📚 <a href="https://huggingface.co/datasets/FreedomIntelligence/CoD-PatientSymDisease" target="_blank">CoD Data</a> • 📚 <a href="https://huggingface.co/datasets/FreedomIntelligence/Disease_Database" target="_blank">Disease Database</a> • 📝 <a href="https://huggingface.co/datasets/FreedomIntelligence/DxBench" target="_blank">DxBench(Benchmark)</a>
</h4>
</div>

<div align="center">
<h4>
  🤗 <a href="https://huggingface.co/FreedomIntelligence/DiagnosisGPT-34B" target="_blank">DiagnosisGPT-34B</a>  • 🤗 <a href="https://huggingface.co/FreedomIntelligence/DiagnosisGPT-6B">DiagnosisGPT-6B</a> 
</h4>
</div>

## ✨ Updates
- [2024/07/19]: We launched our diagnostic LLMs, [DiagnosisGPT-6B](https://huggingface.co/FreedomIntelligence/DiagnosisGPT-6B) and [DiagnosisGPT-34B](https://huggingface.co/FreedomIntelligence/DiagnosisGPT-34B), supporting automatic diagnosis for **9,604** diseases.
- [2024/07/18]: We released our data, including [CoD data](https://huggingface.co/datasets/FreedomIntelligence/CoD-PatientSymDisease), the [disease database](https://huggingface.co/datasets/FreedomIntelligence/Disease_Database), and the [DxBench](https://huggingface.co/datasets/FreedomIntelligence/DxBench) benchmark.
- [2024/07/18]: We introduced the **Chain-of-Diagnosis (CoD)** to improve the trustworthiness and interpretability of medical diagnostics.

## ⚡ Introduction
We propose the Chain-of-Diagnosis to improve interpretability in medical diagnostics for Large Language Models (LLMs). CoD features include:
1. Transforming the opaque decision-making process into a five-step diagnostic chain that reflects a physician’s thought process.
2. Producing a confidence distribution, where higher confidence suggests greater certainty in diagnosing a specific disease. CoD formalizes the diagnostic process as a process of reducing diagnostic certainty entropy.

<div align=center>
<img src="assets/CoD.png" width = "640" alt="HuatuoGPT" align=center/>
</div>


## Citation

```
@misc{chen2024codinterpretablemedicalagent,
      title={CoD, Towards an Interpretable Medical Agent using Chain of Diagnosis}, 
      author={Junying Chen and Chi Gui and Anningzhe Gao and Ke Ji and Xidong Wang and Xiang Wan and Benyou Wang},
      year={2024},
      eprint={2407.13301},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.13301}, 
}
```
