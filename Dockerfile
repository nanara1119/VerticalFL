# 패키지 설치
FROM tensorflow/tensorflow:2.4.1
#pip install tensorflow-gpu==2.3.0
RUN python -m pip install --upgrade pip
RUN pip install sklearn
#RUN pip install scikit-learn==0.23.1
RUN pip install pandas
RUN pip install matplotlib

# 작업 폴더 추가
RUN mkdir script
RUN mkdir results
RUN mkdir data

WORKDIR /script
