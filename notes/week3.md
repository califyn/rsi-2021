fix list 1:

- [x] more Noise
- [x] fix variance
- [x] prefer euclidean over cosine
- [x] cropped image view
- [x] time window
- [x] time range
- [x] min energy max energy
- [x] gap fixes
- [x] test set 20x per energy, less pendulums (maybe 100 ? if possible)
- [ ] only one side against y axis

fix list 2: unfolding wish list
- [ ] maximum variance
- [ ] partially deep NN on output
- [ ] isomap, + correlation (spearman?)
- [ ] kernel regression???
- [ ] pca
- [ ] spectral embedding
- [ ] density

fix list 3: self-sup tactics wish list
- [x] infonce
- [ ] simclr
- [x] simsiam
- [ ] byol
- [ ] normie batch norm
- [ ] decorrelated batch norm
- [ ] barlow twins
- [ ] (and maybe take a look at that vision transformer paper?)

wish list:
- [ ] "objective" battery of tests for noise resistance + partial data + interpolation + extrapolation
  - [ ] include more than one image creation method
- [ ] "objective" criterion for embedding (remember cosine exists too)
- [ ] analyzing noise as a dimension
- [ ] what is an interesting question to ask?
- [ ] WHY do we have compression


does clipping change embedding

noise = temporal?

how much noise can u use

true extrapolation
energy/phase extrapolation/interpolation
only on one side of the y=0 axis

two views are close in time
when doing time restriction like your method, make sure the period is large enough that it doesnt cover the entire thing

stick to 1d lol

spearman correlation test for rankings vs linear regression

spearman correlation vs noise: parabolic/"sweet spot"

spearman correlation vs proportion of the period

spearman correlation vs discontinuities/gaps (interpolation vs. extrapolation)

https://arxiv.org/pdf/2005.10243.pdf infomin principle -- also theres a bunch of citations for frame based video contrastive

https://openreview.net/pdf?id=enoVQWLsfyL viewmaker
-----
well,it seems like the model chooses the "best" possible place to put the blank images.  best being minimizing expected distance to true encoding

does compression happen b/c of clipping? will representations get bigger if clipped weaker?

mathematically does it help to increase the distance?

https://www.pnas.org/content/116/19/9501 lstm for progression

https://arxiv.org/pdf/2102.12525v2.pdf prior image based generative. sort of similar
