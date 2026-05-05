/analysis
    __init__.py
    dial_ablation.py
    domain_correlation.py
    efficiency_curve.py
    eigenvalue_semantics.py
    geometric_audit.py
    layer_profiles.py
    residual_stability.py

/configs - DONE
    /ablations - DONE
        bias_ablation.yaml - DONE
        dial_ablation.yaml - DONE
    /domains - DONE
        biomedical.yaml - DONE
        code.yaml - DONE
        legal.yaml - DONE
        news.yaml - D0NE
    /methods - DONE
        bitfit.yaml - DONE
        frozen.yaml - DONE
        full_finetuning.yaml - DONE
        hybrid_paft.yaml - DONE
        lora_r8.yaml - DONE
        lora_r64.yaml - DONE
        pure_paft.yaml - DONE
        safe_hybrid_paft.yaml - DONE
        safe_pure_paft.yaml - DONE
    /models - DONE
        gpt2_large.yaml - DONE
        gpt2_medium.yaml - DONE
        gpt2_small.yaml - DONE
    base.yaml - DONE

/paft 
    /checkpointing 
        __init__.py
        loader.py
        saver.py
        schema.py
    /data
        __init__.py
        base.py
        biomedical.py
        code.py
        legal.py
        news.py
        utils.py
    /decomposition
        __init__.py
        geometry.py - DONE
        polar.py - DONE
        validators.py
    /methods - DONE
        /baselines - DONE
            bitfit.py - DONE
            frozen.py - DONE
            full_finetune.py - DONE
            lora.py - DONE
            polar.py - DONE
            svf.py - DONE
        __init__.py - DONE
        base.py -  DONE
        hybrid_paft.py - DONE
        pure_paft.py - DONE
        safe_hybrid_paft.py - DONE
        safe_pure_paft.py - DONE
    /metrics 
        __init__.py
        classification.py
        generation.py
        perplexity.py
    /model - DONE
        __init__.py - DONE
        extractor.py - DONE
        paft_model.py - DONE
        parameter_groups.py - DONE
        reconstruction.py - DELETED (Redundent logic with paft_model.py. Moved the "get_OV_circuits" to /analysis/geoemtric_audit.py)
        svf_model.py - DONE
    /training
        __init__.py
        callbacks.py
        landing_field.py
        scheduler.py
        trainer.py
    /utils
        __init__.py
        config.py - DONE
        device.py - DONE
        log_utils.py - DONE
        reproducibility.py

/results

/scripts

/tests

setup.py
        

