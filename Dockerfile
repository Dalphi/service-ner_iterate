FROM python:3

COPY . /usr/src/app
WORKDIR /usr/src/app
RUN \
	wget http://hal3.name/megam/megam_i686.opt.gz && \
	gunzip megam_i686.opt.gz && \
	mv megam_i686.opt /bin/megam && \
	chmod +x /bin/megam && \
	pip3 install -r requirements.txt && \
	python3 -m nltk.downloader all

CMD python3 iterate_service.py
