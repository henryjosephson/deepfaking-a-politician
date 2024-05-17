# Deepfaking a Politician
### A DATA 259 Project by Henry Josephson and Otto Keppke

<!-- add an ai-generated image? could be cute-->

## Background
In the world of politics, [there](https://www.politico.com/newsletters/new-york-playbook/2024/01/23/faked-ai-audio-hits-harlem-politics-00137132) [have](https://www.nbcnews.com/politics/2024-election/fake-joe-biden-robocall-tells-new-hampshire-democrats-not-vote-tuesday-rcna134984) [been](https://www.msn.com/en-in/entertainment/other/ashutosh-rana-opens-up-on-his-deepfake-video-supporting-political-party-it-takes-years-to-build-an-image-just-a-day/ar-BB1mbRvC) [plenty](https://www.npr.org/2023/06/08/1181097435/desantis-campaign-shares-apparent-ai-generated-fake-images-of-trump-and-fauci) [of](https://www.wired.com/story/slovakias-election-deepfakes-show-ai-is-a-danger-to-democracy/) [deepfakes](https://www.cbsnews.com/chicago/news/vallas-campaign-deepfake-video/) [in](https://factcheck.afp.com/doc.afp.com.336Z8Q6) [the](https://www.youtube.com/results?search_query=us+presidents+play+minecraft) [news](https://www.thelocal.dk/20240513/explained-how-ai-deep-fakes-are-bringing-new-tensions-to-danish-politics) [lately](https://www.nytimes.com/2023/10/20/nyregion/ai-robocalls-eric-adams.html). It seems like it's getting easier and easier to make politicians say things that they never actually said, which would obviously be bad -- in the examples linked above, we see what seem to be politicians insulting their allies, making endorsements, doctoring images of their rivals, and so forth.

This seems robustly bad. Particularly in a democracy (or at least a system that aspires to be a democracy), it's important that voters have the truth about the political figures they're electing. Deepfakes don't just produce literal false information; they also probably make people more skeptical of *real* content.

This is all pretty scary stuff — **but how hard is it to actually deepfake a politician?** Is it something any idiot could do in thirty seconds (and is it therefore something we should expect a lot of?), or is it something that takes lots of technical expertise?

In our project, Otto and I set out to answer these questions (and, spoiler alert — the most-convincing deepfake that we found took around <1 hour of effort, basically no technical expertise, and $11.)

## How to reproduce our work
You can break down our work into seven broad steps:

0. Get permission from the person you're deepfaking
1. Get training data
2. Process training data
3. Train cheap, free online deepfake
4. Fine-tune state-of-the-art text-to-speech model.
5. Train leading proprietary deepfake
6. Evaluate the three deepfakes

Let's go through these one-by-one.

### 0. Get permission from the person you're deepfaking
Okay, this one is hopefully straightforward — this started as a personal curiosity that I adapted into a school project — neither Otto nor I are trying to create political propaganda, and we're definitely not trying to run afoul of the law.

It's very convenient, then, that I spent the summer of 2023 as a legislative aide for New York State Assemblymember Alex Bores. AM Bores doesn't just have a master's in machine learning from Georgia Tech — he's particularly worried about the sudden 

So I asked him — do you mind if I deepfake you?
![](assets/IMG_8972.PNG)

Fortunately, he said yes.

### 1. Get training data
We were lucky — this step was incredibly straightforward, since Assemblymember Bores had just recorded an hour-long interview for the Max Politics Podcast, which is freely available online, e.g. on Soundcloud [here](https://soundcloud.com/gotham-gazette-max-murphy/assemblymember-alex-bores-on-ai-opportunity-court-reform-housing-policy-more). We used [this tool](https://soundcloudmp3.org) to download the audio, though any of the myriad comparable Soundcloud-to-mp3 tools would've been just as good.

### 2. Process training data

### 3. Train cheap, free online deepfake

### 4. Fine-tune state-of-the-art text-to-speech model.

### 5. Train leading proprietary deepfake

### 6. Evaluate the three deepfakes