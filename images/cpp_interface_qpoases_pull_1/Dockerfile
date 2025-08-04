FROM juanjqo/ubuntu_dev
SHELL ["/bin/bash", "-c"]
ENV BASH_ENV="/etc/bash_env"

# Install qpOASES
RUN mkdir -p ~/utils && \
cd ~/utils && git clone https://github.com/coin-or/qpOASES.git && \
cd ~/utils/qpOASES && sed -i -e 's/option(BUILD_SHARED_LIBS "If ON, build shared library instead of static" OFF)/option(BUILD_SHARED_LIBS "If ON, build shared library instead of static" ON)/g' CMakeLists.txt && \
cd ~/utils/qpOASES/ && mkdir -p build && \
cd ~/utils/qpOASES/build  && cmake .. && make -j$(nproc) && make install

# Install qpoases interface proposed by qlin960618
RUN mkdir -p ~/utils && \
cd ~/utils && git clone https://github.com/qlin960618/cpp-interface-qpoases.git && \
cd ~/utils/cpp-interface-qpoases && mkdir -p build && \
cd ~/utils/cpp-interface-qpoases/build && cmake .. && make -j$(nproc) && make install

# Install proxsuite
RUN mkdir -p ~/utils && \
cd ~/utils && git clone https://github.com/Simple-Robotics/proxsuite.git --recursive && \
cd ~/utils/proxsuite && mkdir -p build && \
cd ~/utils/proxsuite/build  && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_WITH_VECTORIZATION_SUPPORT=OFF && make -j$(nproc) && make install

# Install proxqp interface
RUN mkdir -p ~/utils && \
cd ~/utils && git clone https://github.com/juanjqo/cpp-interface-proxqp.git && \
cd ~/utils/cpp-interface-proxqp && mkdir -p build && \
cd ~/utils/cpp-interface-proxqp/build && cmake .. && make -j$(nproc) && make install


# Install DQ Robotics for C++ (development branch)
RUN add-apt-repository ppa:dqrobotics-dev/development && apt-get update && apt-get install -y libdqrobotics


# Clone dqrobotics_internal_tests
RUN mkdir -p ~/git && git clone https://github.com/juanjqo/dqrobotics_internal_tests.git

