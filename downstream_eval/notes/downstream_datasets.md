<font size=3>

## Visual Instruction Tuning
https://arxiv.org/pdf/2304.08485.pdf  
This is not a paper about a downstream task so move it to the proper notes directory.  
This paper uses the concept of giving GPT-4 bouding boxes of objects in an image in order to let GPT reason about spatial relations in the image using only language. This is similar to what we are doing for text.

## TextCaps
https://arxiv.org/pdf/2003.12462.pdf  

## ST-VQA
https://arxiv.org/pdf/1905.13648.pdf

## TextVQA
https://arxiv.org/pdf/1904.08920.pdf  

## OCR-VQA (probably ❌)
https://anandmishra22.github.io/files/mishra-OCR-VQA.pdf  
Has only book covers and questions based on the text on them.  

## VizWiz-Captions (probably ❌)
https://arxiv.org/pdf/2002.08565.pdf  
Probably don't want to use this since there is emphasis on not using specific phrases from the text in the image in the captions. This seems to reduce the need for precise OCR (although maybe the models will ultimately benefit from having more accurate OCR text).  

## Hateful Memes (probably ❌) 
https://arxiv.org/pdf/2005.04790v3.pdf  
Don't know if this is suitable since the original paper defines the task of the dataset as classifying memes ***given the text in the image and image*** so it doesn't require OCR. ~~However the CLIP paper (https://arxiv.org/pdf/2103.00020.pdf) classified it as a task requiring OCR.~~  

## OCR-CC  
https://arxiv.org/pdf/2012.04638.pdf  
This paper also has a model trained and evaluated on various vision-language tasks which could be used to evaluate our pipeline.  

## TextPlace Self Collected Dataset + ETH V4RL Urban Place Recognition Dataset + SYNTHIA dataset
(https://openaccess.thecvf.com/content_ICCV_2019/papers/Hong_TextPlace_Visual_Place_Recognition_and_Topological_Localization_Through_Reading_Scene_ICCV_2019_paper.pdf)  
(https://arxiv.org/pdf/2205.08565.pdf)  
Robotics downstream task which uses OCR.

## https://www.cs.cmu.edu/~kaess/pub/Wang15iros.pdf
Another possible downstream robotics task

## http://europa2.informatik.uni-freiburg.de/files/radwan16icra.pdf
Another possible downstream robotics task

</font>