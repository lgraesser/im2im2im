### Datasets

- CelebA
- 128 x 128 x 3
- 20k images in each distribution
- Blond: Blond_hair = 1, Eyeglasses = -1, dataset = 0
- Brunette: Brown_hair = 1, Eyeglasses = -1, dataset = 0
- Smiling: Smiling = 1, Blond_hair = -1, Brown_hair = -1, dataset = 0
- Not smiling: Smiling = -1, Blond_hair = -1, Brown_hair = -1, dataset = 0

### Experiments

1. Jointly trained model, four distributions, pairwise training
    - Large initial kernel: 7 x 7, padding 3
2. Two separately trained models, double loop completed afterwards
    - Small initial kernel: 3 x 3, padding 1
3. Jointly trained model (as in 1.) warm started with two separately trained models
    - Blond - brunette: trained for 100k iterations
    - Smiling - not smiling: trained for 172k iterations (harder distribution to learn)
    - Small initial kernel: 3 x 3, padding 1


### Images Key
  - \*jt.jpg = model 1
  - \*st.jpg = model 2
  - \*.jpg = model 3

### Observations
  - Smiling to not smiling seems easier for brunette distribution: it is possible that this is a biased dataset and it is not possible to exclude mouths whilst training
  - Mode drop in model 1. Not smiling - smiling fails. This is resolved with the warm start for model 3. It is also possible that the kernel size was too big (but that wouldn't explain smiling to not smiling working)
  - Warm start produces the best results, irons out more extreme translations. This makes sense given the two pair wise models don't share a latent space. Clear improvement from 1 --> 2 --> 3e
    - However smiling definitely becomes less extreme
    - The joint model has a clear preference for less smiling (why?)

### To do
  - Test the models on the validation dataset
  - Check the detailed losses in the log files
