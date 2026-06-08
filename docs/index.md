---
layout: default
title: "Quantitative Inductive Machines"
description: "Quantitative Inductive Machines (QIM): Easy sequence-to-sequence regression with language models, for scientific and engineering applications."
---

<style>
    /* --- HEADER STYLES --- */
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
        margin-bottom: 10px;
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
        height: 50vh;
        min-height: 450px;
        max-height: 650px;
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        overflow: hidden;
        width: 100%;
        margin: 30px 0;
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
        flex: 2;
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
        padding: 20px;
    }
    .figure-state.active {
        opacity: 1;
        pointer-events: auto;
    }
    .figure-state img {
        max-width: 100%;
        max-height: 100%;
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
        .figure-col { height: 40vh; min-height: 300px; }
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
        <!-- TODO: Update author list -->
        <b>Author 1</b><sup>1</sup>,
        <b>Author 2</b><sup>1</sup>,
        <b>Author 3</b><sup>1</sup>,
        <b>Author 4</b><sup>1</sup>
    </p>
    <p class="affiliations">
        <sup>1</sup>Google DeepMind
    </p>

    <div class="links">
        <a href="https://arxiv.org/abs/XXXX.XXXXX">📄 Paper</a>
        <a href="https://github.com/google-deepmind/regress-lm">💻 Code</a>
        <a href="https://github.com/google-deepmind/regress-lm/tree/main/colabs">📒 Colabs</a>
    </div>
</div>

## Intro
Given an observation of a complex system, **what number(s) will it produce?**
<img class="teaser-gif" src="assets/intro.jpeg" alt="QIM Introduction">

<!-- Replace with slide table?
This question always arises across science and engineering. For example:

* What accuracy will my ML experiment code reach?
* How many milliseconds will my custom GPU kernel run?
* How efficient is this data center?
* How much plasma is produced by this nuclear fusion reaction?
* What is the survival prognosis for this patient with cancer?
-->

Historically, entire fields have traditionally resorted to _tabular regression_, which represents all worldly information as tables, or precisely, normalized fixed-dimensional vectors. But the world isn't a table, and tabular methods can't be applied to code, logs, or free-form text, which possesses arbitrary _sequence_ lengths.

We instead represent numeric prediction as a **sequence-to-sequence transduction** problem.

## Method Overview
An encoder-decoder converts tokens to tokens, from one information space (the raw observations of the world) into another, the spaces of all real numbers. Inputs $x$ can be represented as-is, and output numbers $y$ can stay unnormalized:

* By using **cross-attention** (instead of compressive embeddings attached to a tabular head), information is preserved and even allows approximating any _computable function._
* By training with **cross-entropy** loss over numeric targets, we smoothly learn any density $p(y \mid x)$ to express epistemic and aleatoric uncertainty properly.
* By applying at scale, we can perform enormous amounts of transfer-learning over any (x,y) data pairs.

At inference, decoding numbers essentially allows us to perform intuitive, or _inductive reasoning_ about the world.

<img class="teaser-gif" src="assets/method_preview.jpeg" alt="Method Preview">

## Applications

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
            <p>Zero expertise needed, still achieve 48% against SoTA</p>
        </div>
        <div class="stage-description" onclick="showStage(3)">
            <h3>GPU Kernel Latency and Optimization</h3>
            <p>16-100x fewer trials needed</p>
        </div>
        <div class="stage-description" onclick="showStage(4)">
            <h3>Static Analysis</h3>
            <p>24+ different languages covered</p>
        </div>
        <div class="stage-description" onclick="showStage(5)">
            <h3>CPU Microarchitecture Simulation</h3>
            <p>Explore</p>
        </div>
        <div class="stage-description" onclick="showStage(6)">
            <h3>TPU Pareto Frontier Generation</h3>
            <p>Pareto Frontiers for TPU Co-Design</p>
        </div>
        <div class="stage-description" onclick="showStage(7)">
            <h3>Data Center Efficiency Prediction</h3>
            <p>From raw telemetry logs</p>
        </div>
        <div class="stage-description" onclick="showStage(8)">
            <h3>Nuclear Fusion Surrogates</h3>
            <p>First to predict from code</p>
        </div>
    </div>
    <div class="figure-col">
        <!-- TODO: Replace placeholder images with actual figures -->
        <div class="figure-state active" onclick="openModal(this)">
            <img src="assets/method_preview.jpeg" alt="Application 1: Performance Prediction">
        </div>
        <div class="figure-state" onclick="openModal(this)">
            <img src="assets/intro.jpeg" alt="Application 2: Code Regression">
        </div>
        <div class="figure-state" onclick="openModal(this)">
            <img src="assets/method_preview.jpeg" alt="Application 3: Scientific Simulation">
        </div>
        <div class="figure-state" onclick="openModal(this)">
            <img src="assets/intro.jpeg" alt="Application 4: Healthcare Prognosis">
        </div>
    </div>
</div>

<!-- Image modal for full-screen view -->
<div id="image-modal" onclick="closeModal()">
    <span id="modal-close">&times;</span>
    <img id="modal-img" src="" alt="Full-size figure">
</div>

## Citation

If you find this work useful, please cite:

<div class="bibtex-box">
<!-- TODO: Update with actual Nature citation -->
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
