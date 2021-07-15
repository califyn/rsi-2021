i think sigmoid output after the linear layer is best

implemented two frame version, maybe will try blur later if this doesnt work

or tomorrow

why is gpu same speed as cpu :|

yay supervised works

![](../data/img_supervised_cve.png)

(on test)

now trying unsupervised!

got gpu to work, it is faster, but god darn who thought it would be funny to
make you type .cuda() everywhere -_-

loss remained at 6.238 for an impressive 290 EPOCHS before going downwards ...

PARAMETERS are conserved quantities

gaps make it sad (takes forever to do better than 6.238)

one that works [0,0.3+0.4,0.7+0.8,1] is faithful:

![](../data/img_self_cve_gaps_tri.png)

with normalizing: WORKS A LOT BETTER trains within 10 epochs:

(output normalized as well)

![](../data/img_self_test10.png)

simsiam just collapses :| or produces very useless things (let's try to fix simsiam tomorrow)

information theory x gaps

this is the normal version with angle vs energy:

![](../data/img_self_slope.png)

why the little tail up? maybe bc hard to distinguish at low energy?

how to tell if an embedding is "good"
