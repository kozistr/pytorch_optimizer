# v3.8.1

## Change Log

### Feature

* Implement `FriendlySAM` optimizer. (#424, #434)
    * [Friendly Sharpness-Aware Minimization](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Friendly_Sharpness-Aware_Minimization_CVPR_2024_paper.pdf) 
* Implement `AdaGO` optimizer. (#436, #437)
    * [AdaGrad Meets Muon: Adaptive Stepsizes for Orthogonal Updates](https://arxiv.org/abs/2509.02981) 
* Update `EXAdam` optimizer to the latest version. (#438)
* Update `EmoNavi` optimizer to the latest version. (#433, #439)
* Implement `Conda` optimizer. (#440, #441)
    * [Conda: Column-Normalized Adam for Training Large Language Models Faster](https://arxiv.org/abs/2509.24218)

### Update

* Accept the `GaloreProjector` parameters in the init params of the `Conda` optimizer. (#443, #444)

### Bug

* Fix NaN problem when grad norm is zero in StableSPAM optimizer. (#431)

### Docs

* Update the documentation page. (#428)

## Contribution

thanks to @liveck, @AhmedMostafa16

# v3.8.0

## Change Log

### Feature

* Implement `EmoNeco` and `EmoZeal` optimizers. (#407)
* Implement `Refined Schedule-Free AdamW` optimizer. (#409, #414)
    * [Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training](https://arxiv.org/abs/2507.09846)
    * You can use this variant by setting `decoupling_c` parameter in the `ScheduleFreeAdamW` optimizer.
* Add more built-in optimizers, `NAdam`, `RMSProp`, and `LBFGS` optimizers. (#415)
* Support `cautious` variant for `Muon` optimizer. (#417)
* Separate distributed functionality from `Muon` to `DistribtuedMuon` optimizer. (#418)
* Implement `StochasticAccumulator`, which is a gradient hook. (#418)
    * [stochastic optimizer](https://github.com/lodestone-rock/torchastic/)

### Update

* Re-implement `Muon` and `AdaMuon` optimizers based on the recent official implementation. (#408, #410)
    * Their definitions have changed from the previous version, so please check out the documentation!
* Update the missing optimizers from `__init__.py`. (#415)
* Add the HuggingFace Trainer example. (#415)
* Optimize the visualization outputs and change the visualization document to a table layout. (#416)

### Dependency

* Update `mkdocs` dependencies. (#417)

### CI

* Add some GitHub actions to automate some processes. (#411, #412, #413)

## Contributions

thanks to @AidinHamedi

# v3.7.0

## Change Log

### Feature

* Implement `AdaMuon` optimizer. (#394, #395)
    * [Adaptive Muon Optimizer](https://arxiv.org/abs/2507.11005v1)
* Implement `SPlus` optimizer. (#396, #399)
    * [A Stable Whitening Optimizer for Efficient Neural Network Training](https://arxiv.org/abs/2506.07254) 
* Implement `EmoNavi`, `EmoFact`, and `EmoLynx` optimizers. (#393, #400)
    * [An emotion-driven optimizer that feels loss and navigates accordingly](https://github.com/muooon/EmoNavi)

### CI

* Enable CI for Python 3.8 ~ 3.13. (#402, #404)

### Fix

* Adjust the value of `eps` to the fixed value `1e-15` when adding to `exp_avg_sq`. (#397, #398)
* built-in type-hint in `Kron` optimizer. (#404)

# v3.6.1

## Change Log

### Feature

* Implement more cooldown types for WSD learning rate scheduler. (#382, #386)
* Implement `AdamWSN` optimizer. (#387, #389)
    * [Lean and Mean Adaptive Optimization via Subset-Norm and Subspace-Momentum with Convergence Guarantees](https://arxiv.org/abs/2411.07120)
* Implement `AdamC` optimizer. (#388, #390)
    * [Why Gradients Rapidly Increase Near the End of Training](https://arxiv.org/abs/2506.02285)

### Update

* Change the default range of the `beta` parameter from `[0, 1]` to `[0, 1)`. (#392)

### Fix

* Fix to use `momentum buffer` instead of the gradient to calculate LMO. (#385)

# v3.6.0

## Change Log

### Feature

* Implement `Fira` optimizer. (#376)
    * [Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?](https://arxiv.org/abs/2410.01623) 
* Implement `RACS` and `Alice` optimizers. (#376)
    * [Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension](https://arxiv.org/abs/2502.07752)
* Implement `VSGD` optimizer. (#377, #378)
    * [Variational Stochastic Gradient Descent for Deep Neural Networks](https://openreview.net/forum?id=xu4ATNjcdy) 
* Enable training with complex parameters. (#370, #380)
    * will raise `NoComplexParameterError` for unsupported optimizers, due to its design or not-yet-implemented.
* Support `maximize` parameter. (#370, #380)
    * `maximize`: maximize the objective with respect to the params, instead of minimizing.
* Implement `copy_stochastic()` method. (#381)

### Update

* Support 2D< Tensor for `RACS` and `Alice` optimizers. (#380)
* Remove the auxiliary variants from the default parameters of the optimizers and change the name of the state and parameter. (#380)
    * `use_gc`, `adanorm`, `cautious`, `stable_adamw`, and `adam_debias` will be affected.
    * You can still use these variants by passing the parameters to `**kwargs`.
    * Notably, in case of `adanorm` variant, you need to pass `adanorm` (and `adanorm_r` for `r` option) parameter(s) to use this variant, and the name of the state will be changed from `exp_avg_norm` to `exp_avg_adanorm`.
* Refactor `reset()` to `init_group()` method in the `BaseOptimizer` class. (#380)
* Refactor `SAM` optimizer family. (#380)
* Gather `AdamP`, `SGDP` things into `pytorch_optimizer.optimizer.adamp.*`. (#381)
    * `pytorch_optimizer.optimizer.sgdp.SGDP` to `pytorch_optimizer.optimizer.adamp.SGDP`
    * `pytorch_optimizer.optimizer.util.projection` to `pytorch_optimizer.optimizer.adamp.projection`
    * `pytorch_optimizer.optimizer.util.cosine_similarity_by_view` to `pytorch_optimizer.optimizer.adamp.cosine_similarity_by_view`
* Remove `channel_view()` and `layer_view()` from `pytorch_optimizer.optimizer.util`. (#381)

### Fix

* Fix shape mismatch issues in the Galore projection for `reverse_std`, `right`, and `full` projection types. (#376)

# v3.5.1

## Change Log

### Feature

* Implement `ScionLight` optimizer. (#369)

### Update

* Update `SCION` optimizer based on the official implementation. (#369)

### Fix

* Correct the learning rate ratio in `Muon` optimizer properly. (#371, #372, #373)

# v3.5.0

## Change Log

### Feature

* Support `StableSPAM` optimizer. (#358, #359)
    * [How to Train in 4-Bit More Stably than 16-Bit Adam](https://arxiv.org/abs/2502.17055?)
* Support `ScheduleFreeWrapper`. (#334, #360)
* Implement `AdaGC` optimizer. (#364, #366)
    * [Improving Training Stability for Large Language Model Pretraining](https://arxiv.org/abs/2502.11034)
* Implement `Simplified-Ademamix` optimizer. (#364, #366)
    * [Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants](https://arxiv.org/abs/2502.02431)
* Support `Ackley` function for testing optimization algorithms.

### Update

* Update Muon optimizer. (#355, #356)
    * support decoupled weight decay.
    * adjust default hyperparameters the same as the original implementation.
    * support adjusted lr from the Moonlight. you can use it by setting `use_adjusted_lr=True`.
* Tune the performance of the coupled Newton iteration method by 5% increase. (#360)
* Update `SCION` optimizer. (#361)
    * add `scale` parameter.
    * update `get_lmo_direction`.

### Fix

* bias_correction2 in ScheduleFreeRAdam optimizer. (#354)
* potential bug in SPAM optimizer. (#365)
* initialize the `z` state within the `step()` of the ScheduleFreeWrapper. (#363, #366)

# v3.4.2

## Change Log

### Feature

* Implement `SCION` optimizer. (#348, #352)
    * [Training Deep Learning Models with Norm-Constrained LMOs](https://arxiv.org/abs/2502.07529)

### Update

* Update ScheduleFreeSGD, AdamW, RAdam optimizers with the latest. (#351, #353)
* Remove `use_palm` variant in ScheduleFree optimizer due to instability. (#353)
* Ranger25 optimizer. (#353)

### Fix

* Remove `weight decouple` parameter in ScheduleFree optimizers. (#351, #353)

### Docs

* Fix `AliG` optimizer visualization. (#350)

## Contributions

thanks to @AidinHamedi, @hatonosuke

# v3.4.1

## Change Log

### Feature

* Support `GCSAM` optimizer. (#343, #344)
    * [Gradient Centralized Sharpness Aware Minimization](https://arxiv.org/abs/2501.11584)
    * you can use it from `SAM` optimizer by setting `use_gc=True`.
* Support `LookSAM` optimizer. (#343, #344)
    * [Towards Efficient and Scalable Sharpness-Aware Minimization](https://arxiv.org/abs/2203.02714)

### Update

* Support alternative precision training for `Shampoo` optimizer. (#339)
* Add more features to and tune `Ranger25` optimizer. (#340)
    * `AGC` + `Lookahead` variants
    * change default beta1, beta2 to 0.95 and 0.98 respectively
* Skip adding `Lookahead` wrapper in case of `Ranger*` optimizers, which already have it in `create_optimizer()`. (#340)
* Improved optimizer visualization. (#345)
* Rename `pytorch_optimizer.optimizer.gc` to `pytorch_optimizer.optimizer.gradient_centralization` to avoid possible conflict with Python built-in function `gc`. (#349)

### Bug

* Fix to update exp_avg_sq after calculating the denominator in `ADOPT` optimizer. (#346, #347)

### Docs

* Update the visualizations. (#340)

## Contributions

thanks to @AidinHamedi

# v3.4.0

## Change Log

### Feature

* Implement `FOCUS` optimizer. (#330, #331)
    * [First Order Concentrated Updating Scheme](https://arxiv.org/abs/2501.12243) 
* Implement `PSGD Kron` optimizer. (#336, #337)
    * [preconditioned stochastic gradient descent w/ Kron pre-conditioner](https://arxiv.org/abs/1512.04202) 
* Implement `EXAdam` optimizer. (#338, #339)
    * [The Power of Adaptive Cross-Moments](https://arxiv.org/abs/2412.20302)

### Update

* Support `OrthoGrad` variant to `Ranger25`. (#332)
  * `Ranger25` optimizer is my experimental-crafted optimizer, which mixes lots of optimizer variants such as `ADOPT` + `AdEMAMix` + `Cautious` + `StableAdamW` + `Adam-Atan2` + `OrthoGrad`.

### Fix

* Add the missing `state` property in `OrthoGrad` optimizer. (#326, #327)
* Add the missing `state_dict`, and `load_state_dict` methods to `TRAC` and `OrthoGrad` optimizers. (#332)
* Skip when the gradient is sparse in `OrthoGrad` optimizer. (#332)
* Support alternative precision training in `SOAP` optimizer. (#333)
* Store SOAP condition matrices as the dtype of their parameters. (#335)

## Contributions

thanks to @Vectorrent, @kylevedder

# v3.3.4

## Change Log

### Feature

* Support `OrthoGrad` feature for `create_optimizer()`. (#324)
* Enhanced flexibility for the `optimizer` parameter in `Lookahead`, `TRAC`, and `OrthoGrad` optimizers. (#324)
    * Now supports both torch.optim.Optimizer instances and classes
    * You can now use `Lookahead` optimizer in two ways.
        * `Lookahead(AdamW(model.parameters(), lr=1e-3), k=5, alpha=0.5)`
        * `Lookahead(AdamW, k=5, alpha=0.5, params=model.parameters())`
* Implement `SPAM` optimizer. (#324)
    * [Spike-Aware Adam with Momentum Reset for Stable LLM Training](https://arxiv.org/abs/2501.06842)
* Implement `TAM`, and `AdaTAM` optimizers. (#325)
    * [Torque-Aware Momentum](https://arxiv.org/abs/2412.18790)

# v3.3.3

## Change Log

### Feature

* Implement `Grams` optimizer. (#317, #318)
    * [Grams: Gradient Descent with Adaptive Momentum Scaling](https://arxiv.org/abs/2412.17107) 
* Support `stable_adamw` variant for `ADOPT` and `AdEMAMix` optimizer. (#321)
    * `optimizer = ADOPT(model.parameters(), ..., stable_adamw=True)`
* Implement an experimental optimizer `Ranger25` (not tested). (#321)
    * mixing `ADOPT + AdEMAMix + StableAdamW + Cautious + RAdam` optimizers.
* Implement `OrthoGrad` optimizer. (#321)
    * [Grokking at the Edge of Numerical Stability](https://arxiv.org/abs/2501.04697)
* Support `Adam-Atan2` feature for `Prodigy` optimizer when `eps` is None. (#321)
    * [Scaling Exponents Across Parameterizations and Optimizers](https://arxiv.org/abs/2407.05872)

# v3.3.2

## Change Log

### Feature

* Implement `SGDSaI` optimizer. (#315, #316)
    * [No More Adam: Learning Rate Scaling at Initialization is All You Need](https://arxiv.org/abs/2412.11768) 

### Bug

* Clone `exp_avg` before calling `apply_cautious` not to mask `exp_avg`. (#316)

# v3.3.1

## Change Log

### Feature

* Support `Cautious` variant to `AdaShift` optimizer. (#310)
* Save the state of the `Lookahead` optimizer too. (#310)
* Implement `APOLLO` optimizer. (#311, #312)
    * [SGD-like Memory, AdamW-level Performance](https://arxiv.org/abs/2412.05270) 
* Rename the `Apollo` (`An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization`) optimizer name to `ApolloDQN` not to overlap with the new optimizer name `APOLLO`. (#312)
* Implement `MARS` optimizer. (#313, #314)
    * [Unleashing the Power of Variance Reduction for Training Large Models](https://arxiv.org/abs/2411.10438)
* Support `Cautious` variant to `MARS` optimizer. (#314)

### Bug

* Fix `bias_correction` in `AdamG` optimizer. (#305, #308)
* Fix a potential bug when loading the state for `Lookahead` optimizer. (#306, #310)

### Docs

* Add more visualizations. (#310, #314)

## Contributions

thanks to @Vectorrent

# v3.3.0

## Change Log

### Feature

* Support `PaLM` variant for `ScheduleFreeAdamW` optimizer. (#286, #288)
    * you can use this feature by setting `use_palm` to `True`.
* Implement `ADOPT` optimizer. (#289, #290)
    * [Modified Adam Can Converge with Any β2 with the Optimal Rate](https://arxiv.org/abs/2411.02853)
* Implement `FTRL` optimizer. (#291)
    * [Follow The Regularized Leader](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)
* Implement `Cautious optimizer` feature. (#294)
    * [Improving Training with One Line of Code](https://arxiv.org/pdf/2411.16085v1)
    * you can use it by setting `cautious=True` for `Lion`, `AdaFactor` and `AdEMAMix` optimizers.
* Improve the stability of `ADOPT` optimizer. (#294)
    * [Note](https://github.com/iShohei220/adopt?tab=readme-ov-file#update-on-nov-22-2024) 
* Support a new projection type `random` for `GaLoreProjector`. (#294)
* Implement `DeMo` optimizer. (#300, #301)
    * [Decoupled Momentum Optimization](https://arxiv.org/abs/2411.19870)
* Implement `Muon` optimizer. (#302)
    * [MomentUm Orthogonalized by Newton-schulz](https://github.com/KellerJordan/Muon)
* Implement `ScheduleFreeRAdam` optimizer. (#304)
* Implement `LaProp` optimizer. (#304)
    * [Separating Momentum and Adaptivity in Adam](https://arxiv.org/abs/2002.04839)
* Support `Cautious` variant to `LaProp`, `AdamP`, `Adopt` optimizers. (#304).

### Refactor

* Big refactoring, removing direct import from `pytorch_optimizer.*`.
    * I removed some methods not to directly import from it from `pytorch_optimzier.*` because they're probably not used frequently and actually not an optimizer rather utils only used for specific optimizers.
    * `pytorch_optimizer.[Shampoo stuff]` -> `pytorch_optimizer.optimizers.shampoo_utils.[Shampoo stuff]`.
        * `shampoo_utils` like `Graft`, `BlockPartitioner`, `PreConditioner`, etc. You can check the details [here](https://github.com/kozistr/pytorch_optimizer/blob/main/pytorch_optimizer/optimizer/shampoo_utils.py).
    * `pytorch_optimizer.GaLoreProjector` -> `pytorch_optimizer.optimizers.galore.GaLoreProjector`.
    * `pytorch_optimizer.gradfilter_ema` -> `pytorch_optimizer.optimizers.grokfast.gradfilter_ema`.
    * `pytorch_optimizer.gradfilter_ma` -> `pytorch_optimizer.optimizers.grokfast.gradfilter_ma`.
    * `pytorch_optimizer.l2_projection` -> `pytorch_optimizer.optimizers.alig.l2_projection`.
    * `pytorch_optimizer.flatten_grad` -> `pytorch_optimizer.optimizers.pcgrad.flatten_grad`.
    * `pytorch_optimizer.un_flatten_grad` -> `pytorch_optimizer.optimizers.pcgrad.un_flatten_grad`.
    * `pytorch_optimizer.reduce_max_except_dim` -> `pytorch_optimizer.optimizers.sm3.reduce_max_except_dim`.
    * `pytorch_optimizer.neuron_norm` -> `pytorch_optimizer.optimizers.nero.neuron_norm`.
    * `pytorch_optimizer.neuron_mean` -> `pytorch_optimizer.optimizers.nero.neuron_mean`.

### Docs

* Add more visualizations. (#297)

### Bug

* Add optimizer parameter to `PolyScheduler` constructor. (#295)

## Contributions

thanks to @tanganke

# v3.2.0

## Change Log

### Feature

* Implement `SOAP` optimizer. (#275)
    * [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321) 
* Support `AdEMAMix` variants. (#276)
    * `bnb_ademamix8bit`, `bnb_ademamix32bit`, `bnb_paged_ademamix8bit`, `bnb_paged_ademamix32bit`
* Support 8/4bit, fp8 optimizers. (#208, #281)
    * `torchao_adamw8bit`, `torchao_adamw4bit`, `torchao_adamwfp8`.
* Support a module-name-level (e.g. `LayerNorm`) weight decay exclusion for `get_optimizer_parameters`. (#282, #283)
* Implement `CPUOffloadOptimizer`, which offloads optimizer to CPU for single-GPU training. (#284)
* Support a regex-based filter for searching names of optimizers, lr schedulers, and loss functions.

### Bug

* Fix `should_grokfast` condition when initialization. (#279, #280)

## Contributions

thanks to @Vectorrent

# v3.1.2

## Change Log

### Feature

* Implement `AdEMAMix` optimizer. (#272)
    * [THE ADEMAMIX OPTIMIZER: BETTER, FASTER, OLDER](https://arxiv.org/pdf/2409.03137) 

### Bug

* Add `**kwargs` to the parameters for dummy placeholder. (#270, #271)

# v3.1.1

## Change Log

### Feature

* Implement `TRAC` optimizer. (#263)
    * [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://arxiv.org/abs/2405.16642)
* Support `AdamW` optimizer via `create_optimizer()`. (#263)
* Implement `AdamG` optimizer. (#264, #265)
    * [Towards Stability of Parameter-free Optimization](https://arxiv.org/abs/2405.04376) 

### Bug

* Handle the optimizers that only take the `model` instead of the parameters in `create_optimizer()`. (#263)
* Move the variable to the same device with the parameter. (#266, #267)

# v3.1.0

## Change Log

### Feature

* Implement `AdaLomo` optimizer. (#258)
    * [Low-memory Optimization with Adaptive Learning Rate](https://arxiv.org/abs/2310.10195) 
* Support `Q-GaLore` optimizer. (#258)
    * [Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients.](https://arxiv.org/abs/2407.08296)
    * you can use by `optimizer = load_optimizer('q_galore_adamw8bit')`
* Support more bnb optimizers. (#258)
    * `bnb_paged_adam8bit`, `bnb_paged_adamw8bit`, `bnb_*_*32bit`.
* Improve `power_iteration()` speed up to 40%. (#259)
* Improve `reg_noise()` (E-MCMC) speed up to 120%. (#260)
* Support `disable_lr_scheduler` parameter for `Ranger21` optimizer to disable built-in learning rate scheduler. (#261)

### Refactor

* Refactor `AdamMini` optimizer. (#258)
* Deprecate optional dependency, `bitsandbytes`. (#258)
* Move `get_rms`, `approximate_sq_grad` functions to `BaseOptimizer` for reusability. (#258)
* Refactor `shampoo_utils.py`. (#259)
* Add `debias`, `debias_adam` methods in `BaseOptimizer`. (#261)
* Refactor to use `BaseOptimizer` only, not inherit multiple classes. (#261)

### Bug

* Fix several bugs in `AdamMini` optimizer. (#257)

## Contributions

thanks to @sdbds

# v3.0.2

## Change Log

### Feature

* Implement `WSD` LR Scheduler. (#247, #248)
  * [Warmup-Stable-Decay LR Scheduler](https://arxiv.org/abs/2404.06395)
* Add more Pytorch built-in lr schedulers. (#248)
* Implement `Kate` optimizer. (#249, #251)
  * [Remove that Square Root: A New Efficient Scale-Invariant Version of AdaGrad](https://arxiv.org/abs/2403.02648) 
* Implement `StableAdamW` optimizer. (#250, #252)
  * [Stable and low-precision training for large-scale vision-language models](https://arxiv.org/abs/2304.13013) 
* Implement `AdamMini` optimizer. (#246, #253)
  * [Use Fewer Learning Rates To Gain More](https://arxiv.org/abs/2406.16793) 

### Refactor

* Refactor `Chebyschev` lr scheduler modules. (#248)
  * Rename `get_chebyshev_lr` to `get_chebyshev_lr_lambda`.
  * Rename `get_chebyshev_schedule` to `get_chebyshev_perm_steps`.
  * Call `get_chebyshev_schedule` function to get `LamdbaLR` scheduler object.
* Refactor with `ScheduleType`. (#248)

# v3.0.1

## Change Log

### Feature

* Implement `FAdam` optimizer. (#241, #242)
  * [Adam is a natural gradient optimizer using diagonal empirical Fisher information](https://arxiv.org/abs/2405.12807)
* Tweak `AdaFactor` optimizer. (#236, #243)
  * support not-using-first-momentum when beta1 is not given
  * default dtype for first momentum to `bfloat16`
  * clip second momentum to 0.999
* Implement `GrokFast` optimizer. (#244, #245)
  * [Accelerated Grokking by Amplifying Slow Gradients](https://arxiv.org/abs/2405.20233)

### Bug

* Wrong typing of reg_noise. (#239, #240)
* Lookahead`s param_groups attribute is not loaded from checkpoint. (#237, #238)

## Contributions

thanks to @michaldyczko

# v3.0.0

The major version is updated! (`v2.12.0` -> `v3.0.0`) (#164)

Many optimizers, learning rate schedulers, and objective functions are in `pytorch-optimizer`.
Currently, `pytorch-optimizer` supports **67 optimizers (+ `bitsandbytes`)**, **11 lr schedulers**, and **13 loss functions**, and reached about 4 ~ 50K downloads / month (peak is 75K downloads / month)!

The reason for updating the major version from `v2` to `v3` is that I think it's a good time to ship the recent implementations (the last update was about 7 months ago) and plan to pivot to new concepts like training utilities while maintaining the original features (e.g. optimizers). 
Also, rich test cases, benchmarks, and examples are on the list!

Finally, thanks for using the `pytorch-optimizer`, and feel free to make any requests :)

## Change Log

### Feature

* Implement `REX` lr scheduler. (#217, #222)
  * [Revisiting Budgeted Training with an Improved Schedule](https://arxiv.org/abs/2107.04197) 
* Implement `Aida` optimizer. (#220, #221)
  * [A DNN Optimizer that Improves over AdaBelief by Suppression of the Adaptive Stepsize Range](https://arxiv.org/abs/2203.13273)
* Implement `WSAM` optimizer. (#213, #216)
  * [Sharpness-Aware Minimization Revisited: Weighted Sharpness as a Regularization Term](https://arxiv.org/abs/2305.15817)
* Implement `GaLore` optimizer. (#224, #228)
  * [Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
* Implement `Adalite` optimizer. (#225, #229)
* Implement `bSAM` optimizer. (#212, #233)
  * [SAM as an Optimal Relaxation of Bayes](https://arxiv.org/abs/2210.01620)
* Implement `Schedule-Free` optimizer. (#230, #233)
  * [Schedule-Free optimizers](https://github.com/facebookresearch/schedule_free)
* Implement `EMCMC`. (#231, #233)
  * [Entropy-MCMC: Sampling from flat basins with ease](https://www.semanticscholar.org/paper/Entropy-MCMC%3A-Sampling-from-Flat-Basins-with-Ease-Li-Zhang/fd95de3f24fc4f955a6fe5719d38d1d06136e0cd) 

### Fix

* Fix SRMM to allow operation beyond memory_length. (#227)

### Dependency

* Drop `Python 3.7` support officially. (#221)
  * Please check the [README](https://github.com/kozistr/pytorch_optimizer?tab=readme-ov-file#getting-started).
* Update `bitsandbytes` to `0.43.0`. (#228)

### Docs

* Add missing parameters in `Ranger21 optimizer` document. (#214, #215)
* Fix `WSAM` optimizer paper link. (#219)

## Diff

* from the previous major version : [2.0.0...3.0.0](https://github.com/kozistr/pytorch_optimizer/compare/v2.0.0...v3.0.0)
* from the previous version: [2.12.0...3.0.0](https://github.com/kozistr/pytorch_optimizer/compare/v2.12.0...v3.0.0)

## Contributions

thanks to @sdbds, @i404788

# v2.12.0

## Change Log

### Feature

* Support `bitsandbytes` optimizer. (#211)
    * now, you can install with `pip3 install pytorch-optimizer[bitsandbytes]`
    * supports 8 bnb optimizers.
        * `bnb_adagrad8bit`, `bnb_adam8bit`, `bnb_adamw8bit`, `bnb_lion8bit`, `bnb_lamb8bit`, `bnb_lars8bit`, `bnb_rmsprop8bit`, `bnb_sgd8bit`.

### Docs

* Introduce `mkdocs` with `material` theme. (#204, #206)
    * documentation : https://pytorch-optimizers.readthedocs.io/en/latest/

### Diff

[2.11.2...2.12.0](https://github.com/kozistr/pytorch_optimizer/compare/v2.11.2...v2.12.0)

# v2.11.2

## Change Log

### Feature

* Implement DAdaptLion optimizer (#203)
  * [Lion with D-Adaptation](https://github.com/facebookresearch/dadaptation/blob/main/dadaptation/dadapt_lion.py)

### Fix

* Fix Lookahead optimizer (#200, #201, #202)
  * When using PyTorch Lightning which expects your optimiser to be a subclass of `Optimizer`.
* Fix default `rectify` to `False` in `AdaBelief` optimizer (#203)

### Test

* Add `DynamicLossScaler` test case

### Docs

* Highlight the code blocks
* Fix pepy badges

### Diff

[2.11.1...2.11.2](https://github.com/kozistr/pytorch_optimizer/compare/v2.11.1...v2.11.2)

## Contributions

thanks to @georg-wolflein

# v2.11.1

## Change Log

### Feature

* Implement Tiger optimizer (#192)
  * [A Tight-fisted Optimizer](https://github.com/bojone/tiger/blob/main/README_en.md)
* Implement CAME optimizer (#196)
  * [Confidence-guided Adaptive Memory Efficient Optimization](https://aclanthology.org/2023.acl-long.243/) 
* Implement loss functions (#198)
  * Tversky Loss : [Tversky loss function for image segmentation using 3D fully convolutional deep networks](https://arxiv.org/abs/1706.05721)
  * Focal Tversky Loss
  * Lovasz Hinge Loss : [The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/abs/1705.08790)

### Diff

[2.11.0...2.11.1](https://github.com/kozistr/pytorch_optimizer/compare/v2.11.0...v2.11.1)

# v2.11.0

## Change Log

### Feature

* Implement PAdam optimizer (#186)
  * [Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks](https://arxiv.org/abs/1806.06763) 
* Implement LOMO optimizer (#188)
  * [Full Parameter Fine-tuning for Large Language Models with Limited Resources](https://arxiv.org/abs/2306.09782) 
* Implement loss functions (#189)
  * BCELoss
  * BCEFocalLoss
  * FocalLoss : [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
  * FocalCosineLoss : [Data-Efficient Deep Learning Method for Image Classification Using Data Augmentation, Focal Cosine Loss, and Ensemble](https://arxiv.org/abs/2007.07805)
  * DiceLoss : [Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations](https://arxiv.org/abs/1707.03237v3)
  * LDAMLoss : [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://arxiv.org/abs/1906.07413)
  * JaccardLoss
  * BiTemperedLogisticLoss : [Robust Bi-Tempered Logistic Loss Based on Bregman Divergences](https://arxiv.org/abs/1906.03361)

### Diff

[2.10.1...2.11.0](https://github.com/kozistr/pytorch_optimizer/compare/v2.10.1...v2.11.0)

# v2.10.1

## Change Log

### Feature

* Implement Prodigy optimizer (#183)
  * [An Expeditiously Adaptive Parameter-Free Learner](https://arxiv.org/abs/2306.06101) 

### Fix

* `perturb` isn't multiplied by `-step_size` in SWATS optimizer. (#179)
* `chebyshev step` has size of `T` while the permutation is `2^T`. (#168, #181) 

### Diff

[2.10.0...2.10.1](https://github.com/kozistr/pytorch_optimizer/compare/v2.10.0...v2.10.1)

# v2.10.0

## Change Log

### Feature

* Implement Amos optimizer (#174)
  * [An Adam-style Optimizer with Adaptive Weight Decay towards Model-Oriented Scale](https://arxiv.org/abs/2210.11693)
* Implement SignSGD optimizer (#176) (thanks to @i404788)
  * [Compressed Optimisation for Non-Convex Problems](https://arxiv.org/abs/1802.04434)
* Implement AdaHessian optimizer (#176) (thanks to @i404788)
  * [An Adaptive Second Order Optimizer for Machine Learning](https://arxiv.org/abs/2006.00719)
* Implement SophiaH optimizer (#173, #176) (thanks to @i404788)
  * [A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342) 
* Implement re-usable functions to compute hessian in `BaseOptimizer` (#176, #177) (thanks to @i404788)
  * two types of distribution are supported (`gaussian`, `rademacher`). 
* Support `AdamD` variant for AdaHessian optimizer (#177)

### Diff

[2.9.1...2.10.0](https://github.com/kozistr/pytorch_optimizer/compare/v2.9.1...v2.10.0)

# v2.9.1

## Change Log

### Fix

* fix weight decay in Ranger21 (#170)

### Diff

[2.9.0...2.9.1](https://github.com/kozistr/pytorch_optimizer/compare/v2.9.0...v2.9.1)

# v2.9.0

## Change Log

### Feature

* Implement AdaMax optimizer (#148)
  * A variant of Adam based on the infinity norm
* Implement Gravity optimizer (#151)
  * [a Kinematic Approach on Optimization in Deep Learning](https://arxiv.org/abs/2101.09192)
* Implement AdaSmooth optimizer (#153)
  * [An Adaptive Learning Rate Method based on Effective Ratio](https://arxiv.org/abs/2204.00825v1)
* Implement SRMM optimizer (#154)
  * [Stochastic regularized majorization-minimization with weakly convex and multi-convex surrogates](https://arxiv.org/abs/2201.01652)
* Implement AvaGrad optimizer (#155) 
  * [Domain-independent Dominance of Adaptive Methods](https://arxiv.org/abs/1912.01823)
* Implement AdaShift optimizer (#157) 
  * [Decorrelation and Convergence of Adaptive Learning Rate Methods](https://arxiv.org/abs/1810.00143v4)
* Upgrade to D-Adaptation v3 (#158, #159)
* Implement AdaDelta optimizer (#160)
  * [An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701v1) 

### Docs

* Fix readthedocs build issue (#156)
* Move citations into table (#156) 

### Refactor

* Refactor validation logic (#149, #150)
* Rename `amsbound`, `amsgrad` terms into `ams_bound` (#149)
* Return gradient instead of the parameter, AGC. (#149)
* Refactor duplicates (e.g. rectified step size, AMSBound, AdamD, AdaNorm, weight decay) into re-usable functions (#150)
* Move `pytorch_optimizer.experimental` under `pytorch_optimizer.*.experimental`

### Diff

[2.8.0...2.9.0](https://github.com/kozistr/pytorch_optimizer/compare/v2.8.0...v2.9.0)

# v2.8.0

## Change Log

### Feature

* Implement A2Grad optimizer (#136)
  * [Optimal Adaptive and Accelerated Stochastic Gradient Descent](https://arxiv.org/abs/1810.00553)
* Implement Accelerated SGD optimizer (#137)
  * [Accelerating Stochastic Gradient Descent For Least Squares Regression](https://arxiv.org/abs/1704.08227)
* Implement Adaptive SGD optimizer (#139)
  * [Adaptive Gradient Descent without Descent](https://arxiv.org/abs/1910.09529)
* Implement SGDW optimizer (#139)
  * [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
* Implement Yogi optimizer (#140)
  * [Adaptive Methods for Nonconvex Optimization](https://papers.nips.cc/paper_files/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html)
* Implement SWATS optimizer (#141)
  * [Improving Generalization Performance by Switching from Adam to SGD](https://arxiv.org/abs/1712.07628) 
* Implement Fromage optimizer (#142)
  * [On the distance between two neural networks and the stability of learning](https://arxiv.org/abs/2002.03432) 
* Implement MSVAG optimizer (#143)
  * [Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients](https://arxiv.org/abs/1705.07774) 
* Implement AdaMod optimizer (#144)
  * [An Adaptive and Momental Bound Method for Stochastic Learning](https://arxiv.org/abs/1910.12249) 
* Implement AggMo optimizer (#145)
  * [Aggregated Momentum: Stability Through Passive Damping](https://arxiv.org/abs/1804.00325)
* Implement QHAdam, QHM optimizers (#146)
  * [Quasi-hyperbolic momentum and Adam for deep learning](https://arxiv.org/abs/1810.06801)
* Implement PID optimizer (#147)
  * [A PID Controller Approach for Stochastic Optimization of Deep Networks](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_PID.pdf) 

### Bug

* Fix `update` in Lion optimizer (#135)
* Fix `momentum_buffer` in SGDP optimizer (#139)

### Diff

[2.7.0...2.8.0](https://github.com/kozistr/pytorch_optimizer/compare/v2.7.0...v2.8.0)

# v2.7.0

## Change Log

### Feature

* Implement `AdaNorm` optimizer (#133)
  * [AdaNorm: Adaptive Gradient Norm Correction based Optimizer for CNNs](https://arxiv.org/abs/2210.06364)
* Implement `RotoGrad` optimizer (#124, #134)
  * [RotoGrad: Gradient Homogenization in Multitask Learning](https://arxiv.org/abs/2103.02631)
* Implement `D-Adapt Adan` optimizer (#134)
* Support `AdaNorm` variant (#133, #134) 
  * AdaBelief
  * AdamP
  * AdamS
  * AdaPNM
  * diffGrad
  * Lamb
  * RAdam
  * Ranger
  * Adan
* Support `AMSGrad` variant (#133, #134)
  * diffGrad
  * AdaFactor
* Support `degenerated_to_sgd` (#133)
  * Ranger
  * Lamb

### Refactor

* Rename `adamd_debias_term` to `adam_debias` (#133)
* Merge the rectified version with the original (#133)
  * diffRGrad + diffGrad -> diffGrad 
  * RaLamb + Lamb -> Lamb
  * now you can simply use with `rectify=True`

### Bug

* Fix `previous_grad` deepcopy issue in Adan optimizer (#134)

### Diff

[2.6.1...2.7.0](https://github.com/kozistr/pytorch_optimizer/compare/v2.6.1...v2.7.0)
