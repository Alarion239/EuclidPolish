from euclid_polish.web.helpers import experiments
ids = ["f200w-0070", "f200w-0040", "f200w-0420", "f200w-0141", "f200w-0325", "f200w-0106", "f200w-0080",
       "f200w-0320", "f200w-0271", "f200w-0260", "f200w-0238", "f200w-0102", "f200w-0210"]
specs = ["universal:member_249", "universal:member_218", "universal:member_227", "production",
         "milestone:member_257@4000000", "milestone:member_257@8000000"]
rec = experiments.run_experiment([("nexus", i) for i in ids], specs,
                                 label="NEXUS galaxies · spirals, mergers, edge-on disks · 257 to 8M",
                                 log=print)
print(rec["id"], rec["status"], rec.get("counts"), rec.get("errors"))
