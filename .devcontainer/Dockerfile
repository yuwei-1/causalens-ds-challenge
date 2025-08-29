FROM astral/uv:python3.12-bookworm-slim

USER root

WORKDIR /app

COPY requirements.txt .


# RUN apt-get update && apt-get install -y git \
#     && rm -rf /var/lib/apt/lists/*


# RUN pip install numpy \
#     && git clone https://github.com/SciTools/cartopy.git \
#     && cd cartopy && python setup.py install \
#     && cd ..
    
RUN pip install uv
RUN uv pip install --system -r requirements.txt