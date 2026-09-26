// Stand-in for the pre-standard <iostream.h>. Borland accepted the friend
// `operator==<T>` of itellis.h without a prior template declaration; the
// forward declarations below give standard compilers one.
#pragma once
#include <iostream>
using namespace std;
template <class T> class IteradorLlista;
template <class T> bool operator==(IteradorLlista<T>&, IteradorLlista<T>&);
