HaxBall Group-2
===================================

Empty starting project for ARL 2023.

A rough overview:
* `Demo-Agent` contains a more sophisticated example and shows how you should structure your code
* `Jiaxin_Yang` is the directoriy for Jiaxin_Yang
* `Common` holds the shared code
* `haxballenv` is the submodule with the actual environment
* `images` is a working directory to generate gifs

Some instructions
-----------------

After cloning, you have to add the submodule to get the HaxBall code:

```console
git submodule update --init
```

Compilation should work in the corresponding directory (e.g. `Jiaxin_Yang`) with the normal procedure:

```console
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8
```

Hopefully, it is now possible to start the agent in the build directory:

```console
./HaxBallAgentGroup-2
```

Use an IDE such as QtCreator to open the full project. Your IDE should be able to open the CMakeLists.txt directly:

```console
qtcreator CMakeLists.txt
```

Qt and QtCreator, CMake
-----------------------

On Mac and Linux, you can most likely use your paket manager for all three.

For Windows, download from the [Qt Website](https://www.qt.io/offline-installers) the offline installer with the version Qt 5.12.x (matching the Eikon).
Disable during the installation your internet to avoid creating a Qt Account.
During the setup, select also MinGW 64 bit to get a running C++ Compiler on your computer.
Select the QtCreator checkbox to install the IDE.

CMake is available at [Kitware's Website](https://cmake.org/download/), just install it.

Now, you can start Qtcreator and open directly a `CMakeLists.txt` as C++ project.
