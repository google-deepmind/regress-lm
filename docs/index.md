---
layout: default
title: "Quantitative Inductive Machines"
description: "Decoding the Quantitative World from Any Observation"
---

<style>
    /* --- HEADER STYLES --- */
    .page-header .btn {
        display: none !important;
    }
    .page-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 4rem 2rem;
    }
    .project-name {
        color: #ffffff;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
        margin-bottom: 10px;
        font-size: 2.8rem;
    }
    .project-tagline {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px 25px;
        margin-top: 20px;
        border-radius: 8px;
        color: #e0e0e0;
        font-size: 1.2rem;
        line-height: 1.6;
        max-width: 750px;
        margin-left: auto;
        margin-right: auto;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(5px);
    }

    /* --- GENERAL PAGE STYLES --- */
    strong, b {
        color: #000000;
        font-weight: 800;
    }
    .container {
        max-width: 900px;
        margin: auto;
        padding: 0 20px;
    }
    img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        max-width: 100%;
        height: auto;
        border-radius: 5px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .text-center { text-align: center; }
    .authors {
        font-size: 1.1em;
        margin: 0 -50px 10px -50px;
        line-height: 1.8;
    }
    .affiliations {
        font-size: 0.95em;
        color: #555;
        margin-bottom: 20px;
    }
    .links {
        margin-top: 20px;
        margin-bottom: 30px;
    }
    .links a {
        margin: 5px 8px;
        font-size: 1em;
        font-weight: bold;
        text-decoration: none;
        padding: 8px 16px;
        border-radius: 5px;
        background-color: #f6f8fa;
        border: 1px solid #d1d5da;
        color: #0366d6;
        display: inline-block;
    }
    .links a:hover {
        background-color: #0366d6;
        color: #fff;
        border-color: #0366d6;
    }
    h2 {
        border-bottom: 1px solid #eaecef;
        padding-bottom: 0.3em;
        margin-top: 40px;
    }
    .teaser-gif {
        max-width: 80%;
        margin: 30px auto;
    }
    .bibtex-box {
        background-color: #f6f8fa;
        border: 1px solid #e1e4e8;
        border-radius: 6px;
        padding: 16px;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 0.85em;
        overflow-x: auto;
        white-space: pre;
        line-height: 1.5;
    }

    /* --- INTERACTIVE SLIDESHOW VIEWER (disco_rl style) --- */
    .static-wrapper {
        display: flex;
        flex-direction: row;
        height: 480px;
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        overflow: hidden;
        width: 90vw;
        max-width: 1300px;
        position: relative;
        left: 50%;
        transform: translateX(-50%);
        margin: 40px auto;
    }
    .text-col {
        flex: 1;
        min-width: 280px;
        max-width: 380px;
        background-color: #f8f9fa;
        border-right: 1px solid #eaeaea;
        overflow-y: auto;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 16px;
    }
    .stage-description {
        padding: 16px 20px;
        border-radius: 8px;
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #ccc;
        cursor: pointer;
        transition: all 0.2s ease-out;
        opacity: 0.7;
    }
    .stage-description:hover {
        opacity: 0.9;
        transform: translateX(3px);
    }
    .stage-description.active-step {
        opacity: 1;
        border-left-color: #0366d6;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transform: scale(1.02);
    }
    .stage-description h3 {
        margin-top: 0;
        margin-bottom: 8px;
        font-size: 1.1rem;
        color: #24292e;
    }
    .stage-description p {
        margin: 0;
        font-size: 0.9rem;
        color: #586069;
        line-height: 1.5;
    }
    .figure-col {
        flex: 3;
        position: relative;
        background-color: #fff;
        overflow: hidden;
    }
    .figure-state {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        opacity: 0;
        transition: opacity 0.4s ease-in-out;
        pointer-events: none;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 15px;
        box-sizing: border-box;
    }
    .figure-state.active {
        opacity: 1;
        pointer-events: auto;
    }
    .figure-state img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        box-shadow: none;
        border-radius: 0;
    }

    /* --- IMAGE MODAL --- */
    #image-modal {
        display: none;
        position: fixed;
        z-index: 1000;
        left: 0; top: 0;
        width: 100%; height: 100%;
        background-color: rgba(0,0,0,0.9);
        backdrop-filter: blur(5px);
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    #image-modal img {
        max-width: 90%;
        max-height: 85vh;
        object-fit: contain;
        border-radius: 4px;
        box-shadow: 0 5px 30px rgba(0,0,0,0.5);
    }
    #modal-close {
        position: absolute;
        top: 20px; right: 30px;
        color: #f1f1f1;
        font-size: 40px;
        font-weight: bold;
        cursor: pointer;
    }
    #modal-close:hover { color: #bbb; }

    @media (max-width: 900px) {
        .static-wrapper {
            flex-direction: column-reverse;
            height: auto;
        }
        .text-col {
            width: 100%; max-width: none;
            flex-direction: row;
            overflow-x: auto;
            padding: 16px;
        }
        .stage-description { min-width: 220px; margin-right: 12px; }
        .figure-col { width: 100%; aspect-ratio: 16 / 9; height: auto; }
    }
</style>

<!-- MathJax for LaTeX rendering -->
<script>
MathJax = { tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']] } };
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<div class="container" markdown="1">

<div class="text-center">
    <p class="authors">
        <span style="white-space: nowrap;">
            <b>Xingyou (Richard) Song</b><sup><b>1</b>,<b>†</b>,<b>*</b></sup>,
            <b>Yash Akhauri</b><sup><b>2</b>,<b>3</b>,<b>†</b>,<b>*</b></sup>,
            <b>Jiyoun (Jen) Ha</b><sup><b>4</b>,<b>5</b>,<b>*</b></sup>,
            <b>Bryan Lewandowski</b><sup><b>4</b>,<b>*</b></sup>,
        </span><br>
        <b>David Smalling</b><sup><b>1</b></sup>,
        <b>Jason Lowe-Power</b><sup><b>4</b></sup>,
        <b>Jonathan Citrin</b><sup><b>1</b></sup>,
        <b>David Lo</b><sup><b>4</b></sup>,
        <b>Rami Cohen</b><sup><b>4</b></sup>,
        <b>Julian Walker</b><sup><b>1</b></sup>,
        <b>Lai Wei</b><sup><b>4</b></sup>,
        <b>Subhashini Venugopalan</b><sup><b>2</b></sup>,
        <b>Mohamed Abdelfattah</b><sup><b>3</b></sup>,
        <b>Cheng-Hsi Lin</b><sup><b>4</b></sup>,
        <b>Bartłomiej Wróblewski</b><sup><b>1</b></sup>,
        <b>Suvinay Subramanian</b><sup><b>4</b></sup>,
        <b>Daiyi Peng</b><sup><b>1</b></sup>,<br>
        <span style="white-space: nowrap;">
            <b>Denny Zhou</b><sup><b>1</b></sup>,
            <b>Ed Chi</b><sup><b>1</b></sup>,
            <b>Quoc Le</b><sup><b>1</b></sup>,
            <b>Jeff Dean</b><sup><b>1</b></sup>,
            <b>Pushmeet Kohli</b><sup><b>1</b></sup>
        </span>
    </p>
    <p class="affiliations">
        <sup>1</sup>Google DeepMind &nbsp;&nbsp;&nbsp;&nbsp;
        <sup>2</sup>Google Research &nbsp;&nbsp;&nbsp;&nbsp;
        <sup>3</sup>Cornell University &nbsp;&nbsp;&nbsp;&nbsp;
        <sup>4</sup>Google &nbsp;&nbsp;&nbsp;&nbsp;
        <sup>5</sup>Stanford University
    </p>
    <p class="author-notes" style="font-size: 0.9em; color: #555; margin-bottom: 20px;">
        <sup><b>†</b></sup>Equal Lead. &nbsp;&nbsp;&nbsp;&nbsp; <sup><b>*</b></sup>Core Independent Contributor.
    </p>

    <div class="links">
        <a href="https://arxiv.org/abs/XXXX.XXXXX">📄 Paper</a>
        <a href="https://github.com/google-deepmind/regress-lm">💻 Code</a>
        <a href="https://github.com/google-deepmind/regress-lm/tree/main/colabs">📒 Colabs</a>
    </div>
</div>

## Intro
Given an observation of a complex system, **what number(s) will it produce?**
<img class="teaser-gif" src="assets/intro.png" alt="QIM Introduction">

<!-- Make it a 10 = 2x5 to express the huge range?
* What accuracy will my ML experiment code reach?
* How many milliseconds will my custom GPU kernel run?
* How efficient is this data center?
* How much plasma is produced by this nuclear fusion reaction?
* What is the survival prognosis for this patient with cancer?
-->

Historically, entire fields have resorted to traditional _tabular regression_ which represents all information as tables, or precisely, normalized fixed-dimensional vectors. But the world isn't a table. Tabular methods can't be applied to data possessing arbitrary _sequence_ lengths, such as code, logs, or free-form text.

We instead represent numeric prediction as a **sequence-to-sequence** problem.

## Method Overview
A compact encoder-decoder converts, or _transduces_, from the space of all observations into another: the space of all real numbers.

<img class="teaser-gif" src="assets/method_preview.png" alt="Method Preview">

By:

* Expressing **token-by-token**, input observations $x$ can be represented as-is, and output numbers $y$ can stay unnormalized.
* Using **cross-attention** (instead of compressive embeddings attached to a tabular head), information is preserved and even allows approximating any _computable function._
* Training with **cross-entropy** loss over numeric targets, we smoothly learn any (possibly multi-objective) density $p(y \mid x)$ to express epistemic and aleatoric uncertainty properly.
* Scaling up and fine-tuning, we can perform enormous amounts of **transfer-learning** over any $(x,y)$ data pairs.

At inference, decoding numbers allows us to perform intuitive, or _inductive reasoning_ about the world.

<img class="teaser-gif" src="assets/computational.png" alt="Computational Approximation and Density Estimation">

## Applications
Across 10 different high-impact scientific and industrial problems spanning experimental design, code execution, healthcare, and physics, each application achieves at least one of:

1. A new predictive capability not previously demonstrated.
2. Outperforms SoTA without domain-specific architecture or feature engineering.
3. Near-perfect simulation with at orders of magnitude lower cost.
4. Unified data scaling: Massive transfer-learning across different tasks.

<!-- Interactive application viewer (disco_rl style) -->
<div class="static-wrapper">
    <div class="text-col">
        <div class="stage-description active-step" onclick="showStage(0)">
            <h3>Predicting ML Experiments from Code</h3>
            <p>Kaggle Experiment Scores</p>
        </div>
        <div class="stage-description" onclick="showStage(1)">
            <h3>Hyperparameter Optimization Reduction</h3>
            <p>Up to 100x fewer experiments needed</p>
        </div>
        <div class="stage-description" onclick="showStage(2)">
            <h3>Simplifying Neural Architecture Search</h3>
            <p>Zero expertise needed, achieve 48% against SoTA</p>
        </div>
        <div class="stage-description" onclick="showStage(3)">
            <h3>GPU Kernel Optimization</h3>
            <p>16-100x fewer trials needed</p>
        </div>
        <div class="stage-description" onclick="showStage(4)">
            <h3>Static Analysis for Memory</h3>
            <p>24+ different languages covered</p>
        </div>
        <div class="stage-description" onclick="showStage(5)">
            <h3>CPU Microarchitecture Simulation</h3>
            <p>Explore $10^{20}$ hardware configurations quickly</p>
        </div>
        <div class="stage-description" onclick="showStage(6)">
            <h3>TPU/LLM Pareto Frontier Generation</h3>
            <p>Latency + throughput tradeoffs for TPU/LLM co-design</p>
        </div>
        <div class="stage-description" onclick="showStage(7)">
            <h3>Data Center Efficiency</h3>
            <p>Prediction from raw telemetry logs</p>
        </div>
        <div class="stage-description" onclick="showStage(8)">
            <h3>Nuclear Fusion Surrogates</h3>
            <p>Novel inputs from raw code and configs</p>
        </div>
        <div class="stage-description" onclick="showStage(9)">
            <h3>Cancer Survival Prediction</h3>
            <p>Combine 9+ modalities into one model</p>
        </div>
    </div>
    <div class="figure-col">
        <!-- TODO: Replace images with powerpoint slides (consistent dimensions) -->
        <div class="figure-state active" onclick="openModal(this)">
            <img src="assets/kaggle.png" alt="Application: ML Experiment Prediction from Code">
        </div>
        <div class="figure-state" onclick="openModal(this)">
            <img src="assets/hpo.png" alt="Application: Hyperparameter Optimization">
        </div>
        <div class="figure-state" onclick="openModal(this)">
            <img src="assets/nas.png" alt="Application: Neural Architecture Search">
        </div>
        <div class="figure-state" onclick="openModal(this)">
            <img src="assets/gpu_kernel.png" alt="Application: GPU Kernel Optimization">
        </div>
        <div class="figure-state" onclick="openModal(this)">
            <img src="assets/static_analysis.png" alt="Application: Static Analysis">
        </div>
        <div class="figure-state" onclick="openModal(this)">
            <img src="assets/cpu.png" alt="Application: CPU Microarchitecture Simulation">
        </div>
        <div class="figure-state" onclick="openModal(this)">
            <img src="assets/pareto.png" alt="Application: Pareto Frontier Prediction">
        </div>
        <div class="figure-state" onclick="openModal(this)">
            <img src="assets/data_center.png" alt="Application: Data Center Efficiency">
        </div>
        <div class="figure-state" onclick="openModal(this)">
            <img src="assets/fusion.png" alt="Application: Nuclear Fusion Surrogates">
        </div>
        <div class="figure-state" onclick="openModal(this)">
            <img src="assets/medical.png" alt="Application: Cancer Survival Prediction">
        </div>
    </div>
</div>

<!-- Image modal for full-screen view -->
<div id="image-modal" onclick="closeModal()">
    <span id="modal-close">&times;</span>
    <img id="modal-img" src="" alt="Full-size figure">
</div>

## Code Availability

Code can be found in the open-source package ([github.com/google-deepmind/regress-lm](https://github.com/google-deepmind/regress-lm)). The default model trains on a single H100 GPU with inputs of up to 32K tokens, and can be further made to run on consumer hardware by using single-layer encoders and decoders.

We provide the following Colabs and pretrained checkpoints for flagship result demos:

* **Synthetic Density:** [synthetic_density_demo.ipynb](https://github.com/google-deepmind/regress-lm/blob/main/colabs/synthetic_density_demo.ipynb).
* **ML Experiments from Code (Kaggle):** [kaggle_demo.ipynb](https://github.com/google-deepmind/regress-lm/blob/main/colabs/kaggle_demo.ipynb).
* **Triton GPU Kernels:** [triton_demo.ipynb](https://github.com/google-deepmind/regress-lm/blob/main/colabs/triton_demo.ipynb).

Pretraining data sources are listed in the paper.

## Acknowledgements

We thank Yutian Chen, Chen Sun, Vinh Tran, Alexander Rush, Michael Brenner, Dara Bahri, Yifeng Lu, Jonathan Lai, and Zhiyu Wei for early feedback, reviewing, and support of the manuscript.

We further thank Chen Liang, Oscar Li, Fred Zhang, Xuezhi Wang, Erik Lin, Esteban Real, Bangding (Jeffrey) Yang, Jarrod Kahn, Yiding Jiang, Samuel Sokota, Yan (Bill) Huang, Victor Reis, Phitchaya Mangpo Phothilimthana, Jörg Bornschein, Tejas Karkhanis, Amir Yazdan Bakhsh, Sami Abu-El-Haija, Erik Lin, Tung Nguyen, Eric Tang, Arissa Wongpanich, Shane Gu, Yingjie Miao, Qiuyi Zhang, Uri Alon, Shao-Hua Sun, Kuang-Huei Lee, Adrian N. Reyes, Zi Wang, Xinyun Chen, Aviral Kumar, Ke Xue, Rong-Xi Tan, Chansoo Lee, Michal Lukasik, Sagi Perel, and Daniel Golovin for relevant discussions.

We finally thank Parthasarathy Ranganathan, Amin Vahdat, Craig Donner, Martin Dixon, Shibl Mourad, Zoubin Ghahramani, and Benoit Schillings for support.

## Citation

If you find this work useful, please cite:

<div class="bibtex-box">
@article{todo,
    title={TODO},
    author={TODO},
    journal={TODO},
    year={TODO}
}
</div>

<br>

---

<p style="text-align: center; color: #999; font-size: 0.85em;">
<b>Disclaimer:</b> This is not an officially supported Google product.
</p>

<script>
function showStage(index) {
    document.querySelectorAll('.stage-description').forEach(function(el, i) {
        el.classList.toggle('active-step', i === index);
    });
    document.querySelectorAll('.figure-state').forEach(function(el, i) {
        el.classList.toggle('active', i === index);
    });
}

function openModal(el) {
    var img = el.querySelector('img');
    var modal = document.getElementById('image-modal');
    document.getElementById('modal-img').src = img.src;
    modal.style.display = 'flex';
}

function closeModal() {
    document.getElementById('image-modal').style.display = 'none';
}

// Auto-rotate through stages
var currentStage = 0;
var totalStages = document.querySelectorAll('.stage-description').length;
var autoRotate = setInterval(function() {
    currentStage = (currentStage + 1) % totalStages;
    showStage(currentStage);
}, 5000);

// Stop auto-rotate on user interaction
document.querySelectorAll('.stage-description').forEach(function(el) {
    el.addEventListener('click', function() { clearInterval(autoRotate); });
});
</script>

</div>
