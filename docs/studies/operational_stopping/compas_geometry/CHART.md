# Scientific figure contract

Question: how much later does full posterior stability trigger, and which
unchanged profile audits accept the resulting COMPAS checkpoints?

Two horizontal paired-dot panels show ten individually labelled catalogues.
Left: training counts at moment and joint-posterior triggers. Right: maximum
centred profile error divided by its tolerance at those two checkpoints, with
the acceptance threshold at one. These are paired development results, not
independent samples. Retain all ten cases, including the uncertain pilot.

Use Matplotlib for the standalone scientific PNG/PDF in this repository. Use
blue circles for the earlier trigger and orange squares for the later trigger,
grey pair connectors, visible labels, and no figure title. The caption carries
scope and interpretation. Inspect the exported PNG at its final size. The JSON
decisions and audits are the numerical source; the table preserves exact counts,
profile errors, offline reference KL and pass/fail information.
