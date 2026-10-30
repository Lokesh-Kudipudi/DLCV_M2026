# Haar-tSVD: One-Step Image Denoising Using Haar Transform and Tensor SVD

## Preliminary Report: Problem Statement and Literature Review

**Lokesh Kudipudi** — lokesh.k23@iiits.in
**Bharghava Reddy Sathi** — bharghavareddy.s23@iiits.in
**Thotham Reddy Palla** — thotamreddy.p23@iiits.in

Indian Institute of Information Technology Sri City, Chittor

> *Conversion note: figures are not embedded; their captions and legible figure text are retained. Page-number back-references in the bibliography (e.g. "pp. 1, 2, 4") are carried over from the original.*

---

## Abstract

Image denoising underpins many downstream vision tasks, yet the strongest recent denoisers are deep networks that need paired training data and GPU resources, and they often degrade on noise they were not trained for. Classical patch-based methods avoid these costs but usually depend on iterative optimisation or on learning a separate transform for every group of similar patches. This preliminary report defines the problem of training-free, one-step denoising and reviews the literature that motivates it. We then summarise Haar-tSVD, a recent method that models patches with a block-circulant representation, captures patch-level correlation with one pair of shared t-SVD bases, and captures group-level correlation with a fixed Haar transform, so that no local basis learning is needed. We state the problem formally, position the method among sparse, low-rank, tensor and learning-based denoisers, and lay out a plan to implement and evaluate it on Kodak, DND, SIDD and PolyU.

---

## 1. Introduction

Images are corrupted by noise during acquisition and transmission, and the loss in quality affects tasks such as feature extraction, tracking and medical diagnosis [10]. With the number of imaging devices growing, denoising has to be not only accurate but also cheap to run.

Deep neural networks currently lead most benchmarks. Residual CNNs [32], activation-free networks [5] and transformers [31] are trained on clean/noisy pairs, and their accuracy can drop when the test noise differs from the training distribution. Collecting real paired data is costly, and training large models is out of reach for many users. A comparison across several real-world datasets [20] shows that classical patch-based methods remain competitive in such settings.

The classical line of work exploits the non-local self-similarity (NLSS) of natural images [2]: similar patches are grouped, filtered jointly in a transform domain, and written back. BM3D [6] is the best-known example. Many later methods replace the fixed transform with a learned one (PCA, SVD, dictionaries) or with low-rank regularisation, which improves quality but requires per-group learning or iterative solvers at test time.

Recently, Kong et al. [22] proposed Haar-tSVD, which removes the per-group learning step. Their key observation is that, under a circulant representation of a group of patches, the dominant eigenvector of the group Gram matrix is the constant vector, which is also the first row of the Haar matrix. A fixed Haar transform can therefore stand in for a learned PCA along the grouping dimension, while a single pair of t-SVD bases shared by all patches handles patch-level structure.

This report is a first step towards implementing that idea. Section 2 states the problem, Section 3 reviews related work, Section 4 summarises the reference method, and Section 5 describes datasets, metrics, and the implementation plan.

---

## 2. Problem Statement

**Formal definition.** Let $\mathbf{X} \in \mathbb{R}^{H \times W \times 3}$ be an unknown clean colour image and let the observation be

$$
\mathbf{Y} = \mathbf{X} + \mathbf{N}, \tag{1}
$$

where $\mathbf{N}$ is zero-mean noise with standard deviation $\sigma$. For synthetic data, $\mathbf{N}$ is additive white Gaussian noise (AWGN); for real camera data, $\sigma$ is an effective noise level that varies with content and channel. Given only $\mathbf{Y}$ (and an estimate of $\sigma$), the task is to compute an estimate $\hat{\mathbf{X}}$ that maximises fidelity to $\mathbf{X}$, measured by PSNR and SSIM.

**Problem statement.** Design and implement an image denoiser that (i) works in *one step*, i.e. each group of similar patches is processed by a single forward transform, thresholding and inverse transform, with no iterative optimisation; (ii) is *training-free*, i.e. it needs no clean/noisy image pairs, no neural-network training and no learning of a separate basis for each patch group; and (iii) exploits both *global* (patch-level, shared by the whole image) and *local* (group-level, among similar patches) correlations using the t-SVD and the Haar transform. The method must be fast, parallelisable, and competitive with BM3D-type and learned denoisers on synthetic and real-noise benchmarks.

**Objectives.**

1. Implement the Haar-tSVD pipeline: patch grouping, global t-SVD projection, Haar group transform, hard thresholding, inverse transform and aggregation.
2. Reproduce the reported behaviour on Kodak, DND, SIDD and PolyU, and compare with CBM3D and a t-SVD baseline with local bases.
3. Measure run time and study how patch size, group size, search window and threshold affect quality.

**Research questions.**

- **RQ1:** Can a fixed Haar transform replace per-group PCA without a noticeable loss in PSNR/SSIM?
- **RQ2:** Can one set of t-SVD bases, learned once, be shared across all patches (and reused across images)?
- **RQ3:** How much faster is the resulting one-step filter than methods that learn local bases or solve iterative problems?

**Scope and constraints.** The work targets sRGB image denoising with the base Haar-tSVD filter. The published paper also proposes an adaptive variant (A-Haar-tSVD) that uses a CNN noise estimator, and an enhancement (RA-Haar-tSVD) that uses a small fully connected network; both involve learned components and are therefore outside the required scope. We treat them as optional extensions and replace the CNN estimator by a classical noise estimator (Section 5). Video and hyperspectral experiments of the paper are also outside scope.

**Why the problem is hard.**

(a) Noise corrupts patch matching, so groups may contain dissimilar patches.
(b) Real noise is signal-dependent and channel-correlated, unlike AWGN.
(c) Thresholding fails when noise coefficients are as large as signal coefficients, which happens at high noise levels.
(d) Speed must be obtained without giving up the adaptivity that learned bases provide.

---

## 3. Literature Review

We group the literature into five threads: patch-based NLSS filtering, sparse and low-rank models, tensor and circulant representations, deep-learning denoisers, and Haar/wavelet shrinkage. Table 1 summarises how representative methods differ on the properties that matter for this problem.

### 3.1. Patch-based NLSS denoising

Buades et al. [2] introduced non-local means, which averages pixels whose surrounding patches look alike. BM3D [6] extended this idea: it stacks similar patches into a 3D group, applies a fixed 3D transform, shrinks the coefficients, and aggregates the results, followed by a second, Wiener-filtering stage. A colour version [7] performs grouping in a luminance–chrominance space. The shared pipeline of grouping, collaborative filtering and aggregation is the framework used by most later classical methods, including Haar-tSVD. Its main weakness is that the fixed transform cannot adapt to each group, and accuracy falls as noise increases because grouping becomes unreliable.

### 3.2. Sparse coding, PCA and low-rank models

To adapt to image content, K-SVD [9] learns an overcomplete dictionary for sparse patch representation. Zhang et al. [34] apply PCA to locally grouped pixels, and the SVD-based method of Guo et al. [12] follows a similar idea; both learn a basis for each group. Weighted nuclear norm minimisation (WNNM) [11] treats a group as a low-rank matrix, and its multi-channel version [28] handles channel-dependent noise. Trilateral weighted sparse coding [30] models real noise with additional weights. These methods often reach high quality but need per-group learning or iterative solvers at test time, and several rely on hand-tuned parameters.

### 3.3. Tensor, t-SVD and circulant representations

Colour images and patch groups are naturally third- or fourth-order tensors, and vectorising them discards structure [18]. HOSVD-based denoising [24] learns a separable basis in each mode. The t-product and t-SVD of Kilmer et al. [15, 16] instead define tensor multiplication through a block-circulant matrix, so that a third-order tensor is diagonalised by an FFT along its third mode and the t-SVD reduces to independent matrix SVDs of the Fourier slices. Zhang and Aeron [35] used it for tensor completion, and Chen et al. [4] exploited local and non-local circulant similarity for denoising. Kong and Yang [19] built a one-step t-SVD denoiser with block-diagonal representation (MSt-SVD) that still learns bases for each group, and later added a green-channel prior [21] that reduces patch-matching cost by matching mainly on the cleaner green channel [13]. These are the direct predecessors of Haar-tSVD.

### 3.4. Deep-learning denoisers

DnCNN [32] showed that residual learning and batch normalisation make CNN denoisers effective, and FFDNet [33] added a noise-level map for flexible noise handling. Burger et al. [3] earlier demonstrated that even plain multilayer perceptrons could compete with BM3D. NAFNet [5] and Restormer [31] push accuracy with simplified convolutional and transformer blocks. Self-supervised methods such as Blind2Unblind [27] remove the need for clean targets. Elad et al. [10] survey this shift. The limitations relevant here are dependence on training data and noise distribution, large training cost (more than 1000 minutes for Restormer, against about 23 minutes for the small networks of the reference paper, as reported in [22]), and reduced robustness on unseen real noise.

### 3.5. Haar transform and wavelet shrinkage

Hard and soft thresholding of transform coefficients [8] is the classical way to suppress noise, and the Haar transform is the simplest orthogonal wavelet, with a fast $O(K \log K)$ algorithm [25]. NLH [14] combines non-local means with a Haar transform for real-world noise. What is new in Haar-tSVD is a justification for using Haar in place of PCA on patch groups, derived from the circulant structure (Section 4).

### 3.6. Gap addressed

Across these threads, methods that adapt well to the data (learned bases, low-rank models, networks) pay for it in per-group computation, test-time iteration, or training data, while fast fixed-transform methods adapt less. Haar-tSVD targets the middle ground: it keeps the patch-level adaptivity of t-SVD, obtained once from the noisy image or from a reused pair of bases, and replaces the group-level learning with a fixed Haar transform that is motivated by the eigenstructure of circulant groups. The reference paper reports that this reaches quality comparable to per-group PCA at lower cost, but also notes that fixed bases are more sensitive to errors in the noise level, and that all patch-based denoisers struggle at severe noise [22]. These two limitations shape our evaluation plan.

---

**Table 1.** Comparison of representative denoisers on the properties relevant to the problem statement. "Group-level transform" refers to the transform applied along the grouping dimension. Entries for the compared methods follow the descriptions in the reviewed literature; the last row is the method to be implemented.

| Method | Family | Representation | Group-level transform | Iterative | Training data |
|---|---|---|---|---|---|
| NLM [2] | Patch-based | Vector / weighted average | None (weighted mean) | No | No |
| BM3D [6] | Transform-domain | 3D group | Fixed (predefined) | No (two stages) | No |
| NLPCA [34] | PCA | Matrix | Learned per group | No | No |
| WNNM [11] / MCWNNM [28] | Low-rank | Matrix | Learned per group (SVD) | Yes | No |
| TWSC [30] | Sparse coding | Matrix | Learned dictionary | Yes | No |
| NLH [14] | Haar-based NLM | Pixel-level | Haar | No | No |
| MSt-SVD [19] | t-SVD | Block diagonal tensor | Learned per group | No | No |
| DnCNN [32], Restormer [31] | Deep network | Learned features | Learned end-to-end | No | Yes (paired) |
| Haar-tSVD [22] | t-SVD + Haar | Circulant tensor | Fixed Haar | No | No |

---

## 4. Reference Method: Haar-tSVD

This section summarises the method of Kong et al. [22] that we intend to implement. Figure 1 shows the pipeline and Algorithm 1 lists the steps.

**Figure 1.** One-step Haar-tSVD filtering. The bases $\mathbf{U}, \mathbf{V}$ are shared by all groups and $\mathbf{U}_{Haar}$ is a fixed matrix, so no basis is learned per group.

*Pipeline (figure text):* Noisy image $\mathbf{Y}$ → Group $K$ similar patches (GCP matching) → Global t-SVD $\mathbf{U}^T * \mathcal{G} * \mathbf{V}$ → Haar $\times_4 \mathbf{U}_{Haar}$ → Hard threshold $\tau$ → Inverse transforms → Aggregate → $\hat{\mathbf{X}}$

**Notation.** A colour patch is $\mathcal{P} \in \mathbb{R}^{ps \times ps \times 3}$, and $K$ similar patches form a group $\mathcal{G} \in \mathbb{R}^{ps \times ps \times 3 \times K}$. The t-product of two tensors is defined through their block-circulant matrices and is computed slice-wise after an FFT along the third mode, and the t-SVD is $\mathcal{A} = \mathcal{U} * \mathcal{S} * \mathcal{V}^T$ [16].

**Step 1: grouping.** For each reference patch, the $K$ closest patches inside a $W \times W$ window are found. To lower cost and improve noise robustness, matching uses the green channel when it is dominant and the channel average otherwise, following the green-channel prior [21]; the paper sets its weight parameter to 1.2.

**Step 2: global patch-level projection.** Under AWGN, the correlation matrices of noisy Fourier-domain patches equal those of clean patches plus $\sigma^2 \mathbf{I}$ for large $K$, so the eigenvectors, and hence the t-SVD bases $\mathbf{U}, \mathbf{V}$, are preserved. The paper therefore learns one pair from all reference patches (*globally-learned*), and also tests reusing a pair learned from a single image on all other images (*globally-reused*), with almost unchanged PSNR/SSIM on CC15, HighISO and SIDD-val (Table I of the paper).

**Step 3: local group-level Haar transform.** Stacking a group $\mathcal{G}$ in circulant form gives a Gram matrix $\mathrm{circ}(\mathcal{G})\,\mathrm{circ}(\mathcal{G})^T$ that is itself circulant. Its dominant eigenpair is

$$
\lambda_{\max} = \left\| \sum_{i=1}^{K} \mathbf{p}_i \right\|^2, \qquad \mathbf{u}_{\max} = \frac{1}{\sqrt{K}} (1, \ldots, 1)^T, \tag{2}
$$

which for $K$ a power of two equals the first row of the Haar matrix. Haar bases are orthogonal and data-independent, so they replace per-group PCA.

**Step 4: one-step filtering.** The coefficients, thresholded coefficients and estimate are

$$
\mathcal{S} = \mathcal{U}^T * \mathcal{G} * \mathcal{V} \times_4 \mathbf{U}_{Haar}, \tag{3}
$$

$$
\mathcal{S}_\tau = \mathcal{S} \cdot \mathbb{1}(|\mathcal{S}| \geq \tau), \tag{4}
$$

$$
\hat{\mathcal{G}} = \mathcal{U} * \mathcal{S}_\tau * \mathcal{V}^T \times_4 \mathbf{U}_{Haar}^{-1}, \tag{5}
$$

with $\tau = \sigma \sqrt{2 \log (c K ps^2)}$, where $c = 3$ for colour images. The patches of $\hat{\mathcal{G}}$ are averaged back into the image.

**Complexity and parallelism.** Per group the cost is $O(K W^2 ps)$ for search, $O(K ps^3)$ for the t-SVD projection and $O(K \log K)$ for the fast Haar transform. Because all groups use the same bases, they can be processed independently; the authors report over $10\times$ speed-up against a serial MATLAB baseline. The default parameters are $ps = 8$, $K = 32$, $W = 18$.

**Algorithm 1: Haar-tSVD (as described in [22])**

```
Require: Noisy image Y, ps, K, W, noise level σ
1: Learn (or load) global t-SVD bases U, V from reference patches
2: for each reference patch do
3:     Find K similar patches in the W-window ⇒ G
4:     S ← Uᵀ ∗ G ∗ V ×₄ U_Haar
5:     Hard-threshold S with τ
6:     Ĝ ← inverse transform of the thresholded S
7:     Aggregate Ĝ into the output image
8: end for
9: return X̂
```

**Reported results (reference values).** Table 2 lists the PSNR/SSIM values the paper reports for the four datasets in our scope. They serve as targets for our implementation, not as our own results.

**Table 2.** PSNR/SSIM reported in [22] (Table IV). Kodak: independent per-channel noise, case 1 with parameters (15, 10, 20) and case 2 with (40, 30, 50).

| Method | Kodak-1 | Kodak-2 | DND | SIDD | PolyU |
|---|---|---|---|---|---|
| CBM3D | 33.51/.893 | 28.90/.780 | 37.73/.934 | 34.74/.922 | 38.69/.970 |
| MSt-SVD | 33.99/.906 | 29.12/.770 | 38.01/.938 | 34.38/.901 | 38.85/.971 |
| Haar-tSVD | 34.13/.910 | 29.16/.780 | 38.11/.939 | 35.05/.914 | 38.78/.969 |
| A-Haar-tSVD | 34.04/.907 | 29.18/.783 | 38.25/.944 | 35.58/.925 | 38.89/.971 |

---

## 5. Datasets, Evaluation and Implementation Plan

### 5.1. Datasets

The four datasets in the problem statement cover synthetic and real noise (Table 3). Details are recalled from the dataset papers and will be confirmed when the data are downloaded.

**Table 3.** Datasets for evaluation.

| Dataset | Noise | Role in this study |
|---|---|---|
| Kodak [17] | Synthetic Gaussian | Clean 24-image colour set; we add channelwise noise ourselves, so the setting is controlled and PSNR against ground truth is exact. Used to validate the algorithm and the noise-level sensitivity. |
| DND [23] | Real (camera) | Real photographs with reference images; the ground truth is withheld, so scores come from the benchmark's online submission. |
| SIDD [1] | Real (smartphone) | Smartphone images with high-quality ground truth; we use the validation split for local evaluation, where noise is stronger and the method is more challenged. |
| PolyU [29] | Real (camera) | Cropped real noisy images with clean references, at moderate noise levels. |

### 5.2. Metrics and baselines

We report PSNR and SSIM [26] averaged over each dataset, and wall-clock time for a $512 \times 512 \times 3$ image. Baselines are CBM3D [7], a t-SVD method with locally learned bases (MSt-SVD [19]), and, if resources allow, a pretrained CNN such as FFDNet [33], for context only.

### 5.3. Implementation plan

The implementation follows Algorithm 1. We plan a vectorised Python/NumPy version, with groups processed in batches and, where useful, in parallel across image tiles; the authors' public MATLAB code [22] serves as a correctness reference. Design choices specific to our training-free scope are:

- **Noise level.** The base method takes $\sigma$ as input. For Kodak, $\sigma$ is known. For real data we will estimate it with a classical estimator (median absolute deviation of the finest Haar wavelet band [8]), instead of the paper's CNN.
- **t-SVD bases.** Learn $\mathbf{U}, \mathbf{V}$ from the eigenvectors of the row/column correlation matrices of the Fourier-domain reference patches, and test both globally-learned and globally-reused variants.
- **Group size.** Use $K = 32$ (a power of two, as the Haar link requires); other powers of two are used in ablations.
- **Threshold.** Start from $\tau = \sigma\sqrt{2\log(cKps^2)}$ and tune by validation on a small held-out set.

### 5.4. Planned experiments

1. Main comparison on all four datasets against the baselines.
2. Ablation of the group transform: Haar versus per-group PCA (RQ1).
3. Ablation of the t-SVD bases: local, global and reused (RQ2).
4. Sensitivity to $ps$, $K$, $W$ and to errors in $\sigma$; the paper indicates that fixed Haar bases are more sensitive to $\sigma$.
5. Runtime profiling of search, projection and transform stages (RQ3).

### 5.5. Risks and open issues

- Without the CNN estimator, robustness to a wrong $\sigma$ on real data may be lower than the paper's adaptive variant; we will quantify this.
- At severe noise (e.g. SIDD, Kodak case 2), patch matching and thresholding are expected to degrade; the paper notes this limitation [22], and multi-scale processing [36] is one possible remedy for future work.
- DND scoring depends on an external server, and SIDD/DND use different protocols, so numbers from different sources must be compared carefully.
- Only the arXiv preprint has been reviewed in detail; details such as exact parameter values will be checked against the released code.

### 5.6. Timeline

| Phase | Content |
|---|---|
| 1 | Data preparation, grouping, and t-SVD bases |
| 2 | Haar transform, thresholding, aggregation; Kodak validation |
| 3 | Real-noise runs on DND, SIDD, PolyU; noise estimation |
| 4 | Ablations, timing, final report |

---

## 6. Conclusion

We defined the problem of training-free, one-step image denoising that uses both global and local correlations, and reviewed the sparse, low-rank, tensor and deep-learning literature that leads to it. Haar-tSVD [22] addresses the problem by combining shared t-SVD bases with a fixed Haar transform justified by the circulant structure of patch groups. The next stage is to implement the method, verify it on Kodak, and evaluate it on DND, SIDD and PolyU, with ablations that test whether the Haar substitution and basis sharing keep quality while reducing computation.

---

## References

[1] Abdelrahman Abdelhamed, Stephen Lin, and Michael S. Brown. A high-quality denoising dataset for smartphone cameras. In *CVPR*, pages 1692–1700, 2018. *(p. 4)*

[2] Antoni Buades, Bartomeu Coll, and Jean-Michel Morel. A review of image denoising algorithms, with a new one. *Multiscale Model. Simul.*, 4(2):490–530, 2005. *(pp. 1, 2, 4)*

[3] Harold C. Burger, Christian J. Schuler, and Stefan Harmeling. Image denoising: Can plain neural networks compete with BM3D? In *CVPR*, pages 2392–2399, 2012. *(p. 3)*

[4] Fei Chen, Xiangyan Zeng, and Meng Wang. Image denoising via local and nonlocal circulant similarity. *J. Vis. Commun. Image Represent.*, 30:117–124, 2015. *(p. 2)*

[5] Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, and Jian Sun. Simple baselines for image restoration. In *ECCV*, pages 17–33, 2022. *(pp. 1, 3)*

[6] Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. Image denoising by sparse 3-D transform-domain collaborative filtering. *IEEE Trans. Image Process.*, 16(8):2080–2095, 2007. *(pp. 1, 2, 4)*

[7] Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. Color image denoising via sparse 3D collaborative filtering with grouping constraint in luminance-chrominance space. In *IEEE Int. Conf. Image Process.*, pages 313–316, 2007. *(pp. 2, 4)*

[8] David L. Donoho and Iain M. Johnstone. Ideal spatial adaptation by wavelet shrinkage. *Biometrika*, 81(3):425–455, 1994. *(pp. 3, 4)*

[9] Michael Elad and Michal Aharon. Image denoising via sparse and redundant representations over learned dictionaries. *IEEE Trans. Image Process.*, 15(12):3736–3745, 2006. *(p. 2)*

[10] Michael Elad, Bahjat Kawar, and Gregory Vaksman. Image denoising: The deep learning revolution and beyond—a survey paper. *SIAM J. Imaging Sci.*, 16(3):1594–1654, 2023. *(pp. 1, 3)*

[11] Shuhang Gu, Lei Zhang, Wangmeng Zuo, and Xiangchu Feng. Weighted nuclear norm minimization with application to image denoising. In *CVPR*, pages 2862–2869, 2014. *(pp. 2, 4)*

[12] Qiang Guo, Caiming Zhang, Yunfeng Zhang, and Hui Liu. An efficient SVD-based method for image denoising. *IEEE TCSVT*, 26(5):868–880, 2016. *(p. 2)*

[13] Shi Guo, Zhetong Liang, and Lei Zhang. Joint denoising and demosaicking with green channel prior for real-world burst images. *IEEE TIP*, 30:6930–6942, 2021. *(p. 2)*

[14] Yingkun Hou, Jun Xu, Mingxia Liu, Guanghai Liu, Li Liu, Fan Zhu, and Ling Shao. NLH: A blind pixel-level non-local method for real-world image denoising. *IEEE TIP*, 29:5121–5135, 2020. *(pp. 3, 4)*

[15] Misha E. Kilmer and Carla D. Martin. Factorization strategies for third-order tensors. *Linear Algebra Appl.*, 435(3):641–658, 2011. *(p. 2)*

[16] Misha E. Kilmer, Karen Braman, Ning Hao, and Randy C. Hoover. Third-order tensors as operators on matrices: A theoretical and computational framework with applications in imaging. *SIAM J. Matrix Anal. Appl.*, 34(1):148–172, 2013. *(pp. 2, 3)*

[17] Kodak. Kodak lossless true color image suite. http://r0k.us/graphics/kodak. *(p. 4)*

[18] Tamara G. Kolda and Brett W. Bader. Tensor decompositions and applications. *SIAM Rev.*, 51(3):455–500, 2009. *(p. 2)*

[19] Zhaoming Kong and Xiaowei Yang. Color image and multispectral image denoising using block diagonal representation. *IEEE Trans. Image Process.*, 28(9):4247–4259, 2019. *(pp. 2, 4)*

[20] Zhaoming Kong, Fangxi Deng, Haomin Zhuang, Xiaowei Yang, Jun Yu, and Lifang He. A comparison of image denoising methods. *arXiv preprint arXiv:2304.08990*, 2023. *(p. 1)*

[21] Zhaoming Kong, Fangxi Deng, and Xiaowei Yang. Image denoising using green channel prior. *IEEE Trans. Image Process.*, 2025. *(pp. 2, 3)*

[22] Zhaoming Kong, Xiaowei Yang, and Jiahuan Zhang. Image denoising using global and local circulant representation. *arXiv preprint arXiv:2512.23569*, 2025. *(pp. 1, 3, 4, 5)*

[23] Tobias Plötz and Stefan Roth. Benchmarking denoising algorithms with real photographs. In *CVPR*, pages 1586–1595, 2017. *(p. 4)*

[24] Ajit Rajwade, Anand Rangarajan, and Arunava Banerjee. Image denoising using the higher order singular value decomposition. *IEEE TPAMI*, 35(4):849–862, 2013. *(p. 2)*

[25] P. R. Roeser and M. E. Jernigan. Fast Haar transform algorithms. *IEEE Trans. Comput.*, 100(2):175–177, 1982. *(p. 3)*

[26] Zhou Wang, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. *IEEE TIP*, 13(4):600–612, 2004. *(p. 4)*

[27] Zejin Wang, Jiazheng Liu, Guoqing Li, and Hua Han. Blind2Unblind: Self-supervised image denoising with visible blind spots. In *CVPR*, pages 2027–2036, 2022. *(p. 3)*

[28] Jun Xu, Lei Zhang, David Zhang, and Xiangchu Feng. Multi-channel weighted nuclear norm minimization for real color image denoising. In *ICCV*, pages 1096–1104, 2017. *(pp. 2, 4)*

[29] Jun Xu, Hui Li, Zhetong Liang, David Zhang, and Lei Zhang. Real-world noisy image denoising: A new benchmark. *arXiv preprint arXiv:1804.02603*, 2018. *(p. 4)*

[30] Jun Xu, Lei Zhang, and David Zhang. A trilateral weighted sparse coding scheme for real-world image denoising. In *ECCV*, pages 20–36, 2018. *(pp. 2, 4)*

[31] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In *CVPR*, pages 5728–5739, 2022. *(pp. 1, 3, 4)*

[32] Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising. *IEEE TIP*, 26(7):3142–3155, 2017. *(pp. 1, 2, 4)*

[33] Kai Zhang, Wangmeng Zuo, and Lei Zhang. FFDNet: Toward a fast and flexible solution for CNN-based image denoising. *IEEE TIP*, 27(9):4608–4622, 2018. *(pp. 2, 4)*

[34] Lei Zhang, Weisheng Dong, David Zhang, and Guangming Shi. Two-stage image denoising by principal component analysis with local pixel grouping. *PR*, 43(4):1531–1549, 2010. *(pp. 2, 4)*

[35] Zemin Zhang and Shuchin Aeron. Exact tensor completion using t-SVD. *IEEE Trans. Signal Process.*, 65(6):1511–1526, 2017. *(p. 2)*

[36] Maria Zontak, Inbar Mosseri, and Michal Irani. Separating signal from noise using patch recurrence across scales. In *CVPR*, pages 1195–1202, 2013. *(p. 5)*
