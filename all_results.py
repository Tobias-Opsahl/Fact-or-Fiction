# This contains results for models mentioned in the paper. It is copy pasted from the autmatically saved history files
results = {
    "chatgpt20":
        {"existence": 0.6, "substitution": 0.9772727272727273, "multi hop": 0.7, "multi claim": 0.8378378378378378,
         "negation": 0.75, "single hop": 0.8125, "total_test_accuracy": 0.79},
    "chatgpt50":
        {"existence": 0.65, "substitution": 0.9473684210526315, "multi hop": 0.8888888888888888,
         "multi claim": 0.7567567567567568, "negation": 0.5625, "single hop": 0.6829268292682927,
         "total_test_accuracy": 0.72},
    "qa_gnn15":  # Direct non-mix, 30min på 7 epochs
        {"existence": 0.5896843725943033, "substitution": 0.8608860075634792, "multi hop": 0.76410998552822,
         "multi claim": 0.740054661402976, "negation": 0.741248097412481, "single hop": 0.7459816303099885,
         "total_test_accuracy": 0.7501382590421414, "average_test_loss": 0.6179841520870089},
    "qa_gnn33":  # relevant, non-mix, 45 min på 9 epochs
        {"existence": 0.8468052347959969, "substitution": 0.8992436520799568, "multi hop": 0.7457790641582248,
         "multi claim": 0.6993622836319465, "negation": 0.8074581430745814, "single hop": 0.7657864523536165,
         "total_test_accuracy": 0.7611989824134499, "average_test_loss": 0.7217213869938355},
    "qa_gnn44":  # Relevant mix, Ca 2 timer på 22 epochs
        {"existence": 0.8337182448036952, "substitution": 0.8433279308481901, "multi hop": 0.7472262421611191,
         "multi claim": 0.7443061038566656, "negation": 0.7960426179604262, "single hop": 0.7907577497129736,
         "total_test_accuracy": 0.7807764627806658, "average_test_loss": 0.5462393408037368},
    "baseline_no_evidence":  # litt over 30min på 5 epochs
        {"existence": 0.6250962278675904, "substitution": 0.9016747703943814, "multi hop": 0.7327544621321755,
         "multi claim": 0.6747646522927422, "negation": 0.6423135464231354, "single hop": 0.6770952927669346,
         "total_test_accuracy": 0.6898573166685101, "average_test_loss": 0.9082970625787155},
    "baseline_evidence_relevant":  # 4 timer på 10 epochs, epoch 7 best
        {"existence": 0.6104695919938414, "substitution": 0.9024851431658563, "multi hop": 0.7703810902074288,
         "multi claim": 0.8445186759793502, "negation": 0.773972602739726, "single hop": 0.8119977037887486,
         "total_test_accuracy": 0.8024554805884305, "average_test_loss": 0.48024983751562744},
    "baseline_evidence_direct":
        {"existence": 0.590454195535027, "substitution": 0.906807131280389, "multi hop": 0.7761698022190062,
         "multi claim": 0.8329790464621926, "negation": 0.745814307458143, "single hop": 0.802382319173364,
         "total_test_accuracy": 0.7963720827342108, "average_test_loss": 0.44227799563681547},
    "baseline_factkg_no_evidence":
        {"single hop": 0.6964, "multi claim": 0.6331, "existence": 0.6184, "multi hop": 0.7006,
         "negation": 0.6362, "total_test_accuracy": 0.6520},
    "baseline_factkg_evidence":
        {"single hop": 0.8323, "multi claim": 0.7768, "existence": 0.8161, "multi hop": 0.6884,
         "negation": 0.7941, "total_test_accuracy": 0.7765}
}

claim_name_to_table_mapping = {
    "single hop": "One-hop",
    "multi claim": "Conjuction",
    "existence": "Existence",
    "multi hop": "Multi-hop",
    "negation": "Negation",
    "total_test_accuracy": "Total"
}

model_name_to_table_mapping = {
    "qa_gnn15": "QA-GNN (direct)",
    "qa_gnn33": "QA-GNN (contextual)",
    "qa_gnn44": "QA-GNN (single-step)",
    "chatgpt20": "ChatGPT (20 question batch)",
    "chatgpt50": "ChatGPT (50 question batch)",
    "baseline_no_evidence": "BERT (no subgraphs)",
    "baseline_evidence_relevant": "BERT (contextual)",
    "baseline_evidence_direct": "BERT (direct)",
    "baseline_factkg_no_evidence": "FactKG Benchmark line (no subgraphs)",
    "baseline_factkg_evidence": "FactKG Benchmark (subgraphs)"
}
