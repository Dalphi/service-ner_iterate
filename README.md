# NER Iterate Service

This software is a [Dalphi](https://github.com/Dalphi/dalphi) iterate and merge service to do named entity annotations on german texts.

It uses a [TIGER](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger.html) trained part of speech (POS) tagger to work with german texts and a MaxEnt classifier for training a model to recognise labels in a given corpus.

The API takes a (partly) annotated corpus (format is described [here](https://github.com/Dalphi/interface-ner_complete#expected-payload)), extracts all the sentences containing annotations, trains a model and applies it to the whole given corpus. The now more complete annotated corpus is send back (request based or async).

The ML functions are based on NLTK, everything else you'll need to use this work is present in this repo.

## Prerequisites installation

You may need download some nltk specific resources by running

```
python3
>>> import nltk
>>> nltk.download()
```

and following the installation instructions.

It could be necessary to provide a binary for [MegaM](https://www.umiacs.umd.edu/~hal/megam).
Simply download or build it and provide its location by an environment variable, e.g. like

```
export MEGAM=~/.local/bin/megam
```

## Usage

### Prepare

Use the *text shaper* to convert a plain text file to a JSON based exchange format (called *raw data*):

	python3 text_shaper.py input.txt output.json

To convert multiple txt files in a directory use

	python3 text_shaper.py -id=/path/to/input/files/ -od=/path/to/output/files/


### Launch

Start the REST service using

	python3 iterate_service.py -v -l

### Dalphi

- Add the two services (Iterate & Merge) in Dalphi's service overview. Check stdout for details on your address. It will probably be something like  
`http://localhost:5001/iterate` for the Iterate service and  
`http://localhost:5001/merge` for the merge service.
- Assign those services in your project's settings.
- Import all the raw data you generated using the text shaper.
- Run an iteration by selecting the annotation document's tab and click the button within the blank slate.
- Annotate! We recommend the *NER_complete* interface for annotation. It's fully compatible to this service and lives free and open [here](https://github.com/Dalphi/interface-ner_complete).