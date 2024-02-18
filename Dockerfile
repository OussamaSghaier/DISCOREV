FROM pytorch/pytorch:latest
LABEL maintainer="Oussama Ben Sghaier"

WORKDIR /app

COPY code_refinement/ code_refinement/
COPY comment_generation/ comment_generation/ 
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["/bin/bash"]
