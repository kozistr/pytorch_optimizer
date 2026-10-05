# Visualization

Compare optimizer paths on the Rastrigin and Rosenbrock functions in the gallery below.
Each plot shows the continuous path, the final position, and the global minimum.
The 512 × 512 JPEG exports use about 30 path markers to keep the plots readable and reduce file sizes.

The script tunes hyperparameters for each optimizer and function with a fixed random seed.
These examples compare paths on two-coordinate objectives.
They do not measure training speed or predict performance on a model.
The script excludes optimizers that require unsupported shapes or update protocols.

## Generate the plots

Run this command from the repository root to provide the optional plotting dependencies:

```shell
uv run --with hyperopt --with matplotlib python -m examples.visualize_optimizers
```

If you already installed the dependencies, run `uv run just visualize`.
The script saves plots in `docs/visualizations/` and skips files that exist.
To regenerate the gallery, remove the JPEG files from that directory before you run the script.

## Optimizer paths

Select a plot to view it at full size.

| Optimizer | Rastrigin | Rosenbrock |
| --- | --- | --- |
| AccSGD | [![AccSGD on Rastrigin](visualizations/rastrigin_AccSGD.jpg)](visualizations/rastrigin_AccSGD.jpg) | [![AccSGD on Rosenbrock](visualizations/rosenbrock_AccSGD.jpg)](visualizations/rosenbrock_AccSGD.jpg) |
| AdaBelief | [![AdaBelief on Rastrigin](visualizations/rastrigin_AdaBelief.jpg)](visualizations/rastrigin_AdaBelief.jpg) | [![AdaBelief on Rosenbrock](visualizations/rosenbrock_AdaBelief.jpg)](visualizations/rosenbrock_AdaBelief.jpg) |
| AdaBound | [![AdaBound on Rastrigin](visualizations/rastrigin_AdaBound.jpg)](visualizations/rastrigin_AdaBound.jpg) | [![AdaBound on Rosenbrock](visualizations/rosenbrock_AdaBound.jpg)](visualizations/rosenbrock_AdaBound.jpg) |
| AdaDelta | [![AdaDelta on Rastrigin](visualizations/rastrigin_AdaDelta.jpg)](visualizations/rastrigin_AdaDelta.jpg) | [![AdaDelta on Rosenbrock](visualizations/rosenbrock_AdaDelta.jpg)](visualizations/rosenbrock_AdaDelta.jpg) |
| AdaFactor | [![AdaFactor on Rastrigin](visualizations/rastrigin_AdaFactor.jpg)](visualizations/rastrigin_AdaFactor.jpg) | [![AdaFactor on Rosenbrock](visualizations/rosenbrock_AdaFactor.jpg)](visualizations/rosenbrock_AdaFactor.jpg) |
| AdaGC | [![AdaGC on Rastrigin](visualizations/rastrigin_AdaGC.jpg)](visualizations/rastrigin_AdaGC.jpg) | [![AdaGC on Rosenbrock](visualizations/rosenbrock_AdaGC.jpg)](visualizations/rosenbrock_AdaGC.jpg) |
| AdaHessian | [![AdaHessian on Rastrigin](visualizations/rastrigin_AdaHessian.jpg)](visualizations/rastrigin_AdaHessian.jpg) | [![AdaHessian on Rosenbrock](visualizations/rosenbrock_AdaHessian.jpg)](visualizations/rosenbrock_AdaHessian.jpg) |
| Adai | [![Adai on Rastrigin](visualizations/rastrigin_Adai.jpg)](visualizations/rastrigin_Adai.jpg) | [![Adai on Rosenbrock](visualizations/rosenbrock_Adai.jpg)](visualizations/rosenbrock_Adai.jpg) |
| Adalite | [![Adalite on Rastrigin](visualizations/rastrigin_Adalite.jpg)](visualizations/rastrigin_Adalite.jpg) | [![Adalite on Rosenbrock](visualizations/rosenbrock_Adalite.jpg)](visualizations/rosenbrock_Adalite.jpg) |
| Adam | [![Adam on Rastrigin](visualizations/rastrigin_Adam.jpg)](visualizations/rastrigin_Adam.jpg) | [![Adam on Rosenbrock](visualizations/rosenbrock_Adam.jpg)](visualizations/rosenbrock_Adam.jpg) |
| AdaMax | [![AdaMax on Rastrigin](visualizations/rastrigin_AdaMax.jpg)](visualizations/rastrigin_AdaMax.jpg) | [![AdaMax on Rosenbrock](visualizations/rosenbrock_AdaMax.jpg)](visualizations/rosenbrock_AdaMax.jpg) |
| AdamG | [![AdamG on Rastrigin](visualizations/rastrigin_AdamG.jpg)](visualizations/rastrigin_AdamG.jpg) | [![AdamG on Rosenbrock](visualizations/rosenbrock_AdamG.jpg)](visualizations/rosenbrock_AdamG.jpg) |
| AdamMini | [![AdamMini on Rastrigin](visualizations/rastrigin_AdamMini.jpg)](visualizations/rastrigin_AdamMini.jpg) | [![AdamMini on Rosenbrock](visualizations/rosenbrock_AdamMini.jpg)](visualizations/rosenbrock_AdamMini.jpg) |
| AdaMod | [![AdaMod on Rastrigin](visualizations/rastrigin_AdaMod.jpg)](visualizations/rastrigin_AdaMod.jpg) | [![AdaMod on Rosenbrock](visualizations/rosenbrock_AdaMod.jpg)](visualizations/rosenbrock_AdaMod.jpg) |
| AdamP | [![AdamP on Rastrigin](visualizations/rastrigin_AdamP.jpg)](visualizations/rastrigin_AdamP.jpg) | [![AdamP on Rosenbrock](visualizations/rosenbrock_AdamP.jpg)](visualizations/rosenbrock_AdamP.jpg) |
| AdamS | [![AdamS on Rastrigin](visualizations/rastrigin_AdamS.jpg)](visualizations/rastrigin_AdamS.jpg) | [![AdamS on Rosenbrock](visualizations/rosenbrock_AdamS.jpg)](visualizations/rosenbrock_AdamS.jpg) |
| AdamW | [![AdamW on Rastrigin](visualizations/rastrigin_AdamW.jpg)](visualizations/rastrigin_AdamW.jpg) | [![AdamW on Rosenbrock](visualizations/rosenbrock_AdamW.jpg)](visualizations/rosenbrock_AdamW.jpg) |
| Adan | [![Adan on Rastrigin](visualizations/rastrigin_Adan.jpg)](visualizations/rastrigin_Adan.jpg) | [![Adan on Rosenbrock](visualizations/rosenbrock_Adan.jpg)](visualizations/rosenbrock_Adan.jpg) |
| AdaNorm | [![AdaNorm on Rastrigin](visualizations/rastrigin_AdaNorm.jpg)](visualizations/rastrigin_AdaNorm.jpg) | [![AdaNorm on Rosenbrock](visualizations/rosenbrock_AdaNorm.jpg)](visualizations/rosenbrock_AdaNorm.jpg) |
| AdaPNM | [![AdaPNM on Rastrigin](visualizations/rastrigin_AdaPNM.jpg)](visualizations/rastrigin_AdaPNM.jpg) | [![AdaPNM on Rosenbrock](visualizations/rosenbrock_AdaPNM.jpg)](visualizations/rosenbrock_AdaPNM.jpg) |
| AdaShift | [![AdaShift on Rastrigin](visualizations/rastrigin_AdaShift.jpg)](visualizations/rastrigin_AdaShift.jpg) | [![AdaShift on Rosenbrock](visualizations/rosenbrock_AdaShift.jpg)](visualizations/rosenbrock_AdaShift.jpg) |
| AdaSmooth | [![AdaSmooth on Rastrigin](visualizations/rastrigin_AdaSmooth.jpg)](visualizations/rastrigin_AdaSmooth.jpg) | [![AdaSmooth on Rosenbrock](visualizations/rosenbrock_AdaSmooth.jpg)](visualizations/rosenbrock_AdaSmooth.jpg) |
| AdaTAM | [![AdaTAM on Rastrigin](visualizations/rastrigin_AdaTAM.jpg)](visualizations/rastrigin_AdaTAM.jpg) | [![AdaTAM on Rosenbrock](visualizations/rosenbrock_AdaTAM.jpg)](visualizations/rosenbrock_AdaTAM.jpg) |
| AdEMAMix | [![AdEMAMix on Rastrigin](visualizations/rastrigin_AdEMAMix.jpg)](visualizations/rastrigin_AdEMAMix.jpg) | [![AdEMAMix on Rosenbrock](visualizations/rosenbrock_AdEMAMix.jpg)](visualizations/rosenbrock_AdEMAMix.jpg) |
| ADOPT | [![ADOPT on Rastrigin](visualizations/rastrigin_ADOPT.jpg)](visualizations/rastrigin_ADOPT.jpg) | [![ADOPT on Rosenbrock](visualizations/rosenbrock_ADOPT.jpg)](visualizations/rosenbrock_ADOPT.jpg) |
| AggMo | [![AggMo on Rastrigin](visualizations/rastrigin_AggMo.jpg)](visualizations/rastrigin_AggMo.jpg) | [![AggMo on Rosenbrock](visualizations/rosenbrock_AggMo.jpg)](visualizations/rosenbrock_AggMo.jpg) |
| Aida | [![Aida on Rastrigin](visualizations/rastrigin_Aida.jpg)](visualizations/rastrigin_Aida.jpg) | [![Aida on Rosenbrock](visualizations/rosenbrock_Aida.jpg)](visualizations/rosenbrock_Aida.jpg) |
| AliG | [![AliG on Rastrigin](visualizations/rastrigin_AliG.jpg)](visualizations/rastrigin_AliG.jpg) | [![AliG on Rosenbrock](visualizations/rosenbrock_AliG.jpg)](visualizations/rosenbrock_AliG.jpg) |
| Amos | [![Amos on Rastrigin](visualizations/rastrigin_Amos.jpg)](visualizations/rastrigin_Amos.jpg) | [![Amos on Rosenbrock](visualizations/rosenbrock_Amos.jpg)](visualizations/rosenbrock_Amos.jpg) |
| Ano | [![Ano on Rastrigin](visualizations/rastrigin_Ano.jpg)](visualizations/rastrigin_Ano.jpg) | [![Ano on Rosenbrock](visualizations/rosenbrock_Ano.jpg)](visualizations/rosenbrock_Ano.jpg) |
| APOLLO | [![APOLLO on Rastrigin](visualizations/rastrigin_APOLLO.jpg)](visualizations/rastrigin_APOLLO.jpg) | [![APOLLO on Rosenbrock](visualizations/rosenbrock_APOLLO.jpg)](visualizations/rosenbrock_APOLLO.jpg) |
| ApolloDQN | [![ApolloDQN on Rastrigin](visualizations/rastrigin_ApolloDQN.jpg)](visualizations/rastrigin_ApolloDQN.jpg) | [![ApolloDQN on Rosenbrock](visualizations/rosenbrock_ApolloDQN.jpg)](visualizations/rosenbrock_ApolloDQN.jpg) |
| ASGD | [![ASGD on Rastrigin](visualizations/rastrigin_ASGD.jpg)](visualizations/rastrigin_ASGD.jpg) | [![ASGD on Rosenbrock](visualizations/rosenbrock_ASGD.jpg)](visualizations/rosenbrock_ASGD.jpg) |
| AvaGrad | [![AvaGrad on Rastrigin](visualizations/rastrigin_AvaGrad.jpg)](visualizations/rastrigin_AvaGrad.jpg) | [![AvaGrad on Rosenbrock](visualizations/rosenbrock_AvaGrad.jpg)](visualizations/rosenbrock_AvaGrad.jpg) |
| BCOS | [![BCOS on Rastrigin](visualizations/rastrigin_BCOS.jpg)](visualizations/rastrigin_BCOS.jpg) | [![BCOS on Rosenbrock](visualizations/rosenbrock_BCOS.jpg)](visualizations/rosenbrock_BCOS.jpg) |
| BSAM | [![BSAM on Rastrigin](visualizations/rastrigin_BSAM.jpg)](visualizations/rastrigin_BSAM.jpg) | [![BSAM on Rosenbrock](visualizations/rosenbrock_BSAM.jpg)](visualizations/rosenbrock_BSAM.jpg) |
| CAME | [![CAME on Rastrigin](visualizations/rastrigin_CAME.jpg)](visualizations/rastrigin_CAME.jpg) | [![CAME on Rosenbrock](visualizations/rosenbrock_CAME.jpg)](visualizations/rosenbrock_CAME.jpg) |
| Conda | [![Conda on Rastrigin](visualizations/rastrigin_Conda.jpg)](visualizations/rastrigin_Conda.jpg) | [![Conda on Rosenbrock](visualizations/rosenbrock_Conda.jpg)](visualizations/rosenbrock_Conda.jpg) |
| DAdaptAdaGrad | [![DAdaptAdaGrad on Rastrigin](visualizations/rastrigin_DAdaptAdaGrad.jpg)](visualizations/rastrigin_DAdaptAdaGrad.jpg) | [![DAdaptAdaGrad on Rosenbrock](visualizations/rosenbrock_DAdaptAdaGrad.jpg)](visualizations/rosenbrock_DAdaptAdaGrad.jpg) |
| DAdaptAdam | [![DAdaptAdam on Rastrigin](visualizations/rastrigin_DAdaptAdam.jpg)](visualizations/rastrigin_DAdaptAdam.jpg) | [![DAdaptAdam on Rosenbrock](visualizations/rosenbrock_DAdaptAdam.jpg)](visualizations/rosenbrock_DAdaptAdam.jpg) |
| DAdaptAdan | [![DAdaptAdan on Rastrigin](visualizations/rastrigin_DAdaptAdan.jpg)](visualizations/rastrigin_DAdaptAdan.jpg) | [![DAdaptAdan on Rosenbrock](visualizations/rosenbrock_DAdaptAdan.jpg)](visualizations/rosenbrock_DAdaptAdan.jpg) |
| DAdaptLion | [![DAdaptLion on Rastrigin](visualizations/rastrigin_DAdaptLion.jpg)](visualizations/rastrigin_DAdaptLion.jpg) | [![DAdaptLion on Rosenbrock](visualizations/rosenbrock_DAdaptLion.jpg)](visualizations/rosenbrock_DAdaptLion.jpg) |
| DAdaptSGD | [![DAdaptSGD on Rastrigin](visualizations/rastrigin_DAdaptSGD.jpg)](visualizations/rastrigin_DAdaptSGD.jpg) | [![DAdaptSGD on Rosenbrock](visualizations/rosenbrock_DAdaptSGD.jpg)](visualizations/rosenbrock_DAdaptSGD.jpg) |
| DiffGrad | [![DiffGrad on Rastrigin](visualizations/rastrigin_DiffGrad.jpg)](visualizations/rastrigin_DiffGrad.jpg) | [![DiffGrad on Rosenbrock](visualizations/rosenbrock_DiffGrad.jpg)](visualizations/rosenbrock_DiffGrad.jpg) |
| DualAdam | [![DualAdam on Rastrigin](visualizations/rastrigin_DualAdam.jpg)](visualizations/rastrigin_DualAdam.jpg) | [![DualAdam on Rosenbrock](visualizations/rosenbrock_DualAdam.jpg)](visualizations/rosenbrock_DualAdam.jpg) |
| EmoFact | [![EmoFact on Rastrigin](visualizations/rastrigin_EmoFact.jpg)](visualizations/rastrigin_EmoFact.jpg) | [![EmoFact on Rosenbrock](visualizations/rosenbrock_EmoFact.jpg)](visualizations/rosenbrock_EmoFact.jpg) |
| EmoLynx | [![EmoLynx on Rastrigin](visualizations/rastrigin_EmoLynx.jpg)](visualizations/rastrigin_EmoLynx.jpg) | [![EmoLynx on Rosenbrock](visualizations/rosenbrock_EmoLynx.jpg)](visualizations/rosenbrock_EmoLynx.jpg) |
| EmoNavi | [![EmoNavi on Rastrigin](visualizations/rastrigin_EmoNavi.jpg)](visualizations/rastrigin_EmoNavi.jpg) | [![EmoNavi on Rosenbrock](visualizations/rosenbrock_EmoNavi.jpg)](visualizations/rosenbrock_EmoNavi.jpg) |
| EXAdam | [![EXAdam on Rastrigin](visualizations/rastrigin_EXAdam.jpg)](visualizations/rastrigin_EXAdam.jpg) | [![EXAdam on Rosenbrock](visualizations/rosenbrock_EXAdam.jpg)](visualizations/rosenbrock_EXAdam.jpg) |
| FAdam | [![FAdam on Rastrigin](visualizations/rastrigin_FAdam.jpg)](visualizations/rastrigin_FAdam.jpg) | [![FAdam on Rosenbrock](visualizations/rosenbrock_FAdam.jpg)](visualizations/rosenbrock_FAdam.jpg) |
| Fira | [![Fira on Rastrigin](visualizations/rastrigin_Fira.jpg)](visualizations/rastrigin_Fira.jpg) | [![Fira on Rosenbrock](visualizations/rosenbrock_Fira.jpg)](visualizations/rosenbrock_Fira.jpg) |
| FlashAdamW | [![FlashAdamW on Rastrigin](visualizations/rastrigin_FlashAdamW.jpg)](visualizations/rastrigin_FlashAdamW.jpg) | [![FlashAdamW on Rosenbrock](visualizations/rosenbrock_FlashAdamW.jpg)](visualizations/rosenbrock_FlashAdamW.jpg) |
| FOCUS | [![FOCUS on Rastrigin](visualizations/rastrigin_FOCUS.jpg)](visualizations/rastrigin_FOCUS.jpg) | [![FOCUS on Rosenbrock](visualizations/rosenbrock_FOCUS.jpg)](visualizations/rosenbrock_FOCUS.jpg) |
| Fromage | [![Fromage on Rastrigin](visualizations/rastrigin_Fromage.jpg)](visualizations/rastrigin_Fromage.jpg) | [![Fromage on Rosenbrock](visualizations/rosenbrock_Fromage.jpg)](visualizations/rosenbrock_Fromage.jpg) |
| FTRL | [![FTRL on Rastrigin](visualizations/rastrigin_FTRL.jpg)](visualizations/rastrigin_FTRL.jpg) | [![FTRL on Rosenbrock](visualizations/rosenbrock_FTRL.jpg)](visualizations/rosenbrock_FTRL.jpg) |
| GaLore | [![GaLore on Rastrigin](visualizations/rastrigin_GaLore.jpg)](visualizations/rastrigin_GaLore.jpg) | [![GaLore on Rosenbrock](visualizations/rosenbrock_GaLore.jpg)](visualizations/rosenbrock_GaLore.jpg) |
| Grams | [![Grams on Rastrigin](visualizations/rastrigin_Grams.jpg)](visualizations/rastrigin_Grams.jpg) | [![Grams on Rosenbrock](visualizations/rosenbrock_Grams.jpg)](visualizations/rosenbrock_Grams.jpg) |
| Gravity | [![Gravity on Rastrigin](visualizations/rastrigin_Gravity.jpg)](visualizations/rastrigin_Gravity.jpg) | [![Gravity on Rosenbrock](visualizations/rosenbrock_Gravity.jpg)](visualizations/rosenbrock_Gravity.jpg) |
| GrokFastAdamW | [![GrokFastAdamW on Rastrigin](visualizations/rastrigin_GrokFastAdamW.jpg)](visualizations/rastrigin_GrokFastAdamW.jpg) | [![GrokFastAdamW on Rosenbrock](visualizations/rosenbrock_GrokFastAdamW.jpg)](visualizations/rosenbrock_GrokFastAdamW.jpg) |
| Kate | [![Kate on Rastrigin](visualizations/rastrigin_Kate.jpg)](visualizations/rastrigin_Kate.jpg) | [![Kate on Rosenbrock](visualizations/rosenbrock_Kate.jpg)](visualizations/rosenbrock_Kate.jpg) |
| Kron | [![Kron on Rastrigin](visualizations/rastrigin_Kron.jpg)](visualizations/rastrigin_Kron.jpg) | [![Kron on Rosenbrock](visualizations/rosenbrock_Kron.jpg)](visualizations/rosenbrock_Kron.jpg) |
| Lamb | [![Lamb on Rastrigin](visualizations/rastrigin_Lamb.jpg)](visualizations/rastrigin_Lamb.jpg) | [![Lamb on Rosenbrock](visualizations/rosenbrock_Lamb.jpg)](visualizations/rosenbrock_Lamb.jpg) |
| LaProp | [![LaProp on Rastrigin](visualizations/rastrigin_LaProp.jpg)](visualizations/rastrigin_LaProp.jpg) | [![LaProp on Rosenbrock](visualizations/rosenbrock_LaProp.jpg)](visualizations/rosenbrock_LaProp.jpg) |
| LARS | [![LARS on Rastrigin](visualizations/rastrigin_LARS.jpg)](visualizations/rastrigin_LARS.jpg) | [![LARS on Rosenbrock](visualizations/rosenbrock_LARS.jpg)](visualizations/rosenbrock_LARS.jpg) |
| Lion | [![Lion on Rastrigin](visualizations/rastrigin_Lion.jpg)](visualizations/rastrigin_Lion.jpg) | [![Lion on Rosenbrock](visualizations/rosenbrock_Lion.jpg)](visualizations/rosenbrock_Lion.jpg) |
| LoRARite | [![LoRARite on Rastrigin](visualizations/rastrigin_LoRARite.jpg)](visualizations/rastrigin_LoRARite.jpg) | [![LoRARite on Rosenbrock](visualizations/rosenbrock_LoRARite.jpg)](visualizations/rosenbrock_LoRARite.jpg) |
| MADGRAD | [![MADGRAD on Rastrigin](visualizations/rastrigin_MADGRAD.jpg)](visualizations/rastrigin_MADGRAD.jpg) | [![MADGRAD on Rosenbrock](visualizations/rosenbrock_MADGRAD.jpg)](visualizations/rosenbrock_MADGRAD.jpg) |
| Magma | [![Magma on Rastrigin](visualizations/rastrigin_Magma.jpg)](visualizations/rastrigin_Magma.jpg) | [![Magma on Rosenbrock](visualizations/rosenbrock_Magma.jpg)](visualizations/rosenbrock_Magma.jpg) |
| MARS | [![MARS on Rastrigin](visualizations/rastrigin_MARS.jpg)](visualizations/rastrigin_MARS.jpg) | [![MARS on Rosenbrock](visualizations/rosenbrock_MARS.jpg)](visualizations/rosenbrock_MARS.jpg) |
| MSVAG | [![MSVAG on Rastrigin](visualizations/rastrigin_MSVAG.jpg)](visualizations/rastrigin_MSVAG.jpg) | [![MSVAG on Rosenbrock](visualizations/rosenbrock_MSVAG.jpg)](visualizations/rosenbrock_MSVAG.jpg) |
| NAdam | [![NAdam on Rastrigin](visualizations/rastrigin_NAdam.jpg)](visualizations/rastrigin_NAdam.jpg) | [![NAdam on Rosenbrock](visualizations/rosenbrock_NAdam.jpg)](visualizations/rosenbrock_NAdam.jpg) |
| Nero | [![Nero on Rastrigin](visualizations/rastrigin_Nero.jpg)](visualizations/rastrigin_Nero.jpg) | [![Nero on Rosenbrock](visualizations/rosenbrock_Nero.jpg)](visualizations/rosenbrock_Nero.jpg) |
| NovoGrad | [![NovoGrad on Rastrigin](visualizations/rastrigin_NovoGrad.jpg)](visualizations/rastrigin_NovoGrad.jpg) | [![NovoGrad on Rosenbrock](visualizations/rosenbrock_NovoGrad.jpg)](visualizations/rosenbrock_NovoGrad.jpg) |
| PAdam | [![PAdam on Rastrigin](visualizations/rastrigin_PAdam.jpg)](visualizations/rastrigin_PAdam.jpg) | [![PAdam on Rosenbrock](visualizations/rosenbrock_PAdam.jpg)](visualizations/rosenbrock_PAdam.jpg) |
| PID | [![PID on Rastrigin](visualizations/rastrigin_PID.jpg)](visualizations/rastrigin_PID.jpg) | [![PID on Rosenbrock](visualizations/rosenbrock_PID.jpg)](visualizations/rosenbrock_PID.jpg) |
| PNM | [![PNM on Rastrigin](visualizations/rastrigin_PNM.jpg)](visualizations/rastrigin_PNM.jpg) | [![PNM on Rosenbrock](visualizations/rosenbrock_PNM.jpg)](visualizations/rosenbrock_PNM.jpg) |
| Prodigy | [![Prodigy on Rastrigin](visualizations/rastrigin_Prodigy.jpg)](visualizations/rastrigin_Prodigy.jpg) | [![Prodigy on Rosenbrock](visualizations/rosenbrock_Prodigy.jpg)](visualizations/rosenbrock_Prodigy.jpg) |
| QHAdam | [![QHAdam on Rastrigin](visualizations/rastrigin_QHAdam.jpg)](visualizations/rastrigin_QHAdam.jpg) | [![QHAdam on Rosenbrock](visualizations/rosenbrock_QHAdam.jpg)](visualizations/rosenbrock_QHAdam.jpg) |
| QHM | [![QHM on Rastrigin](visualizations/rastrigin_QHM.jpg)](visualizations/rastrigin_QHM.jpg) | [![QHM on Rosenbrock](visualizations/rosenbrock_QHM.jpg)](visualizations/rosenbrock_QHM.jpg) |
| RACS | [![RACS on Rastrigin](visualizations/rastrigin_RACS.jpg)](visualizations/rastrigin_RACS.jpg) | [![RACS on Rosenbrock](visualizations/rosenbrock_RACS.jpg)](visualizations/rosenbrock_RACS.jpg) |
| RAdam | [![RAdam on Rastrigin](visualizations/rastrigin_RAdam.jpg)](visualizations/rastrigin_RAdam.jpg) | [![RAdam on Rosenbrock](visualizations/rosenbrock_RAdam.jpg)](visualizations/rosenbrock_RAdam.jpg) |
| Ranger | [![Ranger on Rastrigin](visualizations/rastrigin_Ranger.jpg)](visualizations/rastrigin_Ranger.jpg) | [![Ranger on Rosenbrock](visualizations/rosenbrock_Ranger.jpg)](visualizations/rosenbrock_Ranger.jpg) |
| Ranger21 | [![Ranger21 on Rastrigin](visualizations/rastrigin_Ranger21.jpg)](visualizations/rastrigin_Ranger21.jpg) | [![Ranger21 on Rosenbrock](visualizations/rosenbrock_Ranger21.jpg)](visualizations/rosenbrock_Ranger21.jpg) |
| Ranger25 | [![Ranger25 on Rastrigin](visualizations/rastrigin_Ranger25.jpg)](visualizations/rastrigin_Ranger25.jpg) | [![Ranger25 on Rosenbrock](visualizations/rosenbrock_Ranger25.jpg)](visualizations/rosenbrock_Ranger25.jpg) |
| RMSprop | [![RMSprop on Rastrigin](visualizations/rastrigin_RMSprop.jpg)](visualizations/rastrigin_RMSprop.jpg) | [![RMSprop on Rosenbrock](visualizations/rosenbrock_RMSprop.jpg)](visualizations/rosenbrock_RMSprop.jpg) |
| ROSE | [![ROSE on Rastrigin](visualizations/rastrigin_ROSE.jpg)](visualizations/rastrigin_ROSE.jpg) | [![ROSE on Rosenbrock](visualizations/rosenbrock_ROSE.jpg)](visualizations/rosenbrock_ROSE.jpg) |
| SaRA | [![SaRA on Rastrigin](visualizations/rastrigin_SaRA.jpg)](visualizations/rastrigin_SaRA.jpg) | [![SaRA on Rosenbrock](visualizations/rosenbrock_SaRA.jpg)](visualizations/rosenbrock_SaRA.jpg) |
| ScalableShampoo | [![ScalableShampoo on Rastrigin](visualizations/rastrigin_ScalableShampoo.jpg)](visualizations/rastrigin_ScalableShampoo.jpg) | [![ScalableShampoo on Rosenbrock](visualizations/rosenbrock_ScalableShampoo.jpg)](visualizations/rosenbrock_ScalableShampoo.jpg) |
| ScheduleFreeAdamW | [![ScheduleFreeAdamW on Rastrigin](visualizations/rastrigin_ScheduleFreeAdamW.jpg)](visualizations/rastrigin_ScheduleFreeAdamW.jpg) | [![ScheduleFreeAdamW on Rosenbrock](visualizations/rosenbrock_ScheduleFreeAdamW.jpg)](visualizations/rosenbrock_ScheduleFreeAdamW.jpg) |
| ScheduleFreeRAdam | [![ScheduleFreeRAdam on Rastrigin](visualizations/rastrigin_ScheduleFreeRAdam.jpg)](visualizations/rastrigin_ScheduleFreeRAdam.jpg) | [![ScheduleFreeRAdam on Rosenbrock](visualizations/rosenbrock_ScheduleFreeRAdam.jpg)](visualizations/rosenbrock_ScheduleFreeRAdam.jpg) |
| ScheduleFreeSGD | [![ScheduleFreeSGD on Rastrigin](visualizations/rastrigin_ScheduleFreeSGD.jpg)](visualizations/rastrigin_ScheduleFreeSGD.jpg) | [![ScheduleFreeSGD on Rosenbrock](visualizations/rosenbrock_ScheduleFreeSGD.jpg)](visualizations/rosenbrock_ScheduleFreeSGD.jpg) |
| SCION | [![SCION on Rastrigin](visualizations/rastrigin_SCION.jpg)](visualizations/rastrigin_SCION.jpg) | [![SCION on Rosenbrock](visualizations/rosenbrock_SCION.jpg)](visualizations/rosenbrock_SCION.jpg) |
| SCIONLight | [![SCIONLight on Rastrigin](visualizations/rastrigin_SCIONLight.jpg)](visualizations/rastrigin_SCIONLight.jpg) | [![SCIONLight on Rosenbrock](visualizations/rosenbrock_SCIONLight.jpg)](visualizations/rosenbrock_SCIONLight.jpg) |
| SGD | [![SGD on Rastrigin](visualizations/rastrigin_SGD.jpg)](visualizations/rastrigin_SGD.jpg) | [![SGD on Rosenbrock](visualizations/rosenbrock_SGD.jpg)](visualizations/rosenbrock_SGD.jpg) |
| SGDP | [![SGDP on Rastrigin](visualizations/rastrigin_SGDP.jpg)](visualizations/rastrigin_SGDP.jpg) | [![SGDP on Rosenbrock](visualizations/rosenbrock_SGDP.jpg)](visualizations/rosenbrock_SGDP.jpg) |
| SGDSaI | [![SGDSaI on Rastrigin](visualizations/rastrigin_SGDSaI.jpg)](visualizations/rastrigin_SGDSaI.jpg) | [![SGDSaI on Rosenbrock](visualizations/rosenbrock_SGDSaI.jpg)](visualizations/rosenbrock_SGDSaI.jpg) |
| SGDW | [![SGDW on Rastrigin](visualizations/rastrigin_SGDW.jpg)](visualizations/rastrigin_SGDW.jpg) | [![SGDW on Rosenbrock](visualizations/rosenbrock_SGDW.jpg)](visualizations/rosenbrock_SGDW.jpg) |
| Shampoo | [![Shampoo on Rastrigin](visualizations/rastrigin_Shampoo.jpg)](visualizations/rastrigin_Shampoo.jpg) | [![Shampoo on Rosenbrock](visualizations/rosenbrock_Shampoo.jpg)](visualizations/rosenbrock_Shampoo.jpg) |
| SignSGD | [![SignSGD on Rastrigin](visualizations/rastrigin_SignSGD.jpg)](visualizations/rastrigin_SignSGD.jpg) | [![SignSGD on Rosenbrock](visualizations/rosenbrock_SignSGD.jpg)](visualizations/rosenbrock_SignSGD.jpg) |
| SimplifiedAdEMAMix | [![SimplifiedAdEMAMix on Rastrigin](visualizations/rastrigin_SimplifiedAdEMAMix.jpg)](visualizations/rastrigin_SimplifiedAdEMAMix.jpg) | [![SimplifiedAdEMAMix on Rosenbrock](visualizations/rosenbrock_SimplifiedAdEMAMix.jpg)](visualizations/rosenbrock_SimplifiedAdEMAMix.jpg) |
| SM3 | [![SM3 on Rastrigin](visualizations/rastrigin_SM3.jpg)](visualizations/rastrigin_SM3.jpg) | [![SM3 on Rosenbrock](visualizations/rosenbrock_SM3.jpg)](visualizations/rosenbrock_SM3.jpg) |
| SOAP | [![SOAP on Rastrigin](visualizations/rastrigin_SOAP.jpg)](visualizations/rastrigin_SOAP.jpg) | [![SOAP on Rosenbrock](visualizations/rosenbrock_SOAP.jpg)](visualizations/rosenbrock_SOAP.jpg) |
| SophiaH | [![SophiaH on Rastrigin](visualizations/rastrigin_SophiaH.jpg)](visualizations/rastrigin_SophiaH.jpg) | [![SophiaH on Rosenbrock](visualizations/rosenbrock_SophiaH.jpg)](visualizations/rosenbrock_SophiaH.jpg) |
| SPAM | [![SPAM on Rastrigin](visualizations/rastrigin_SPAM.jpg)](visualizations/rastrigin_SPAM.jpg) | [![SPAM on Rosenbrock](visualizations/rosenbrock_SPAM.jpg)](visualizations/rosenbrock_SPAM.jpg) |
| SRMM | [![SRMM on Rastrigin](visualizations/rastrigin_SRMM.jpg)](visualizations/rastrigin_SRMM.jpg) | [![SRMM on Rosenbrock](visualizations/rosenbrock_SRMM.jpg)](visualizations/rosenbrock_SRMM.jpg) |
| StableAdamW | [![StableAdamW on Rastrigin](visualizations/rastrigin_StableAdamW.jpg)](visualizations/rastrigin_StableAdamW.jpg) | [![StableAdamW on Rosenbrock](visualizations/rosenbrock_StableAdamW.jpg)](visualizations/rosenbrock_StableAdamW.jpg) |
| StableSPAM | [![StableSPAM on Rastrigin](visualizations/rastrigin_StableSPAM.jpg)](visualizations/rastrigin_StableSPAM.jpg) | [![StableSPAM on Rosenbrock](visualizations/rosenbrock_StableSPAM.jpg)](visualizations/rosenbrock_StableSPAM.jpg) |
| SWATS | [![SWATS on Rastrigin](visualizations/rastrigin_SWATS.jpg)](visualizations/rastrigin_SWATS.jpg) | [![SWATS on Rosenbrock](visualizations/rosenbrock_SWATS.jpg)](visualizations/rosenbrock_SWATS.jpg) |
| TAM | [![TAM on Rastrigin](visualizations/rastrigin_TAM.jpg)](visualizations/rastrigin_TAM.jpg) | [![TAM on Rosenbrock](visualizations/rosenbrock_TAM.jpg)](visualizations/rosenbrock_TAM.jpg) |
| Tiger | [![Tiger on Rastrigin](visualizations/rastrigin_Tiger.jpg)](visualizations/rastrigin_Tiger.jpg) | [![Tiger on Rosenbrock](visualizations/rosenbrock_Tiger.jpg)](visualizations/rosenbrock_Tiger.jpg) |
| VSGD | [![VSGD on Rastrigin](visualizations/rastrigin_VSGD.jpg)](visualizations/rastrigin_VSGD.jpg) | [![VSGD on Rosenbrock](visualizations/rosenbrock_VSGD.jpg)](visualizations/rosenbrock_VSGD.jpg) |
| Yogi | [![Yogi on Rastrigin](visualizations/rastrigin_Yogi.jpg)](visualizations/rastrigin_Yogi.jpg) | [![Yogi on Rosenbrock](visualizations/rosenbrock_Yogi.jpg)](visualizations/rosenbrock_Yogi.jpg) |
