CXXFLAGS = -I include  -std=c++11 -O3 $(shell python3-config --cflags)
LDFLAGS = $(shell python3-config --ldflags)

DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp

LIB_SO = adaptor.so

INCLUDES = -I /usr/include/python2.7

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(CXX_SOURCES) --shared -fPIC

clean:
	rm -rf $(LIB_SO)
