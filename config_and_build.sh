cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=On -S . -B build
cmake --build build -j $(nproc)
