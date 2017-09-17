FROM python:3

COPY . /usr/src/app
WORKDIR /usr/src/app
RUN \
	apt-get update && \
	apt-get install -y ocaml-nox && \
	cd /usr/lib/ocaml && \
	ln -s libcamlstr.a libstr.a && \
	cd /usr/src/app && \
	wget http://hal3.name/megam/megam_src.tgz && \
	tar -xzf megam_src.tgz && \
	cd megam_0.92 && \
	make && \
	mv megam /bin/megam && \
	cd .. && \
	rm -r megam_* && \
	pip3 install -r requirements.txt && \
	python3 -m nltk.downloader all

CMD python3 iterate_service.py
