# NER Iterate Service

This software is in an early development state. It aims to be a [Dalphi](https://github.com/Dalphi/dalphi) iterate service to do NER based annotations.

This repo contains all the tools you need (currently it's based on some NLTK libs to do the ML - so this is a dependency):

- Dalphi iterate and merge services (HTTP)  
  `python3 iterate_service.py -v -l`
- text shaper to convert a plain text file to a JSON based exchange format
  `python3 text_shaper.py input.txt output.json`
- some Python libs for converting the text exchange format to NLTK's tree format and vice versa, handling HTTP communication and a NER pipeline (to easily exchange NLTK's MaxEnt-classifier)

## Usage

### Prepare

Convert your plain text to Dalphi *raw data*

`python3 text_shaper.py input.txt output.json`

### Launch

`python3 iterate_service.py -v -l`

### Dalphi

- Add the two services (Iterate & Merge) in Dalphi's service overview. Check stdout for details on your address. It will probably be something like  
`http://localhost:5001/iterate` for the Iterate service and  
`http://localhost:5001/merge` for the merge service.
- Assign those services in your project's settings.
- Import all the raw data you generated using the text shaper.
- Run an iteration by selecting the annotation document's tab and click the button within the blank slate.
- Annotate : )

## Read

Articles / Blogs / Papers - everything that helps the current development:

- [NLTK book, chaper 7 (Extracting Information from Text)](http://www.nltk.org/book/ch07.html#sec-ner)
- [Lifting the Hood on NLTK's NE Chunker](http://mattshomepage.com/articles/2016/May/23/nltk_nec/)
- [some MaxEnt background](http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)