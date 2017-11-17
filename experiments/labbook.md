### General notes / thoughts

- 3 * 64 * 64 works well with out of the box celeba settings for blondbrunette but not eyeglasses. Perhaps the kernels and / or padding need to be smaller because glasses take up fewer pixels?

- 12-14 November: Experimented with other pairwise translations
  - Hat / no Hat: Didn't work very well, hats too diverse. Perhaps the dataset of 8k images for each set is too mall.
  - Necktie / no necktie: didn't work, necktie is often cropped out of the picture
  - Blond - brunette: works really nicely
  - Smiling - not smiling: starting to work but may benefit from a smaller kernels
  - Eyeglasses - no eyeglasses: same story as Smiling
  - Gender: male / female: This didn't work well but perhaps not for the same reasons as eyeglasses / smiling. Gender is a global feature but significantly more complex than hair. May need significantly more data to yield good results. Smaller kernel will likely help too.

- 14 November: Experimented with shrinking the kernel of the first convolution for the generator and discriminator from 7 x 7 to 3 x 3. This really seemed to help with small elements in the picture
  - Tried for smiling / no smiling, blond - brunette, eyeglasses - no eyeglasses, hat / no hat, necktie - no necktie
  - Improved results for smiling / not smiling, and eyeglasses / no eyeglasses
  - Also an improvement (but harder to tell) for blond - blond
  - All results folders have the suffix \_smallk

- 16 November: First two experiments jointly training two pairs of distributions in the same model. Surprisingly this seems to be working.
  - Two four way pairs were tried: blond, brunette, smiling, not smiling, and blond, brunette, eyeglasses, no Eyeglasses
  - The models were trained jointly from scratch, but the model was only trained to generate / discriminate between each pair (marginal distribution) separately. At evaluation, the double cycle way tested
  - Blond - brunette, smiling - not smiling works best.
    - This experiment also shows some nice characteristics of both generating elements. Generation seems to be a little like, "add more smile", or "make hair darker" rather than "make brunette", or "make smiling". For half smiles, cycling the images through smiling - no smiling decreases the smile, whilst cycling the images through no smiling to smiling increases the smile. Similarly for light brown hair.
    - No change is made to images with large smiles (which makes sense, and is reassuring, since it suggests the generation process can generate the identify function).
    - Overall it seems as if the model has learnt a pretty good concept of hair color and smiling-ness
  - Removing eyeglasses and then translating the hair color works pretty well, but the mode of adding eyeglasses then changing the hair color seems to be lost. This could be because there are no eyeglasses in the blond brunette dataset (which would be problematic since the model should ignore what is on the eyes), but at the moment seems to be a failure of the no eyeglasses - eyeglasses generation. Perhaps this loss could be tweaked to try and boost this part of the network.




### Todo

[x] Smiling / not smiling with out of the box settings
[x] Male / female with out of the box settings
[x] Try reduce the kernel size and / or padding of the network.
[] Train higher resolution (128 * 128) models
[] Try the double loop on separately trained models
[] Try warm starting the models
[] Compared the difference between 1. separately trained models, 2. jointly trained from scratch, and 3. Warm started with a separately trained model. Hold number of iterations and dataset constant.
[] Tweak no eyeglasses --> eyeglasses loss to improve eyeglasses generation in four way model
