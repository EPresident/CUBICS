#pragma once

#include <cassert>

#include <utils/Utils.h>

#define VECTOR_INITIAL_CAPACITY 32

template<typename T>
struct Vector
{
    int size;
    int capacity;
    T* data;

    void initialize(int initialCapacity = VECTOR_INITIAL_CAPACITY)
    {
        size = 0;
        capacity = initialCapacity;
        MemUtils::malloc<T>(&data, initialCapacity);
    }

    void initialize(Vector<T>* other)
    {
        initialize(other->size);
        copy(other);
    }

    void deinitialize()
    {
        MemUtils::free<T>(data);
    }

    inline T& operator[](int index)
    {
        assert(index < size);

        return *(data + index);
    }

    const inline T& operator[](int index) const
    {
        assert(index < size);

        return *(data + index);
    }

    inline T& at(int index)
    {
        assert(index < size);

        return *(data + index);
    }

    const inline T& at(int index) const
    {
        assert(index < size);

        return *(data + index);
    }

    void copy(Vector<T>* other)
    {
        if (capacity < other->size)
        {
            capacity = other->size;
            MemUtils::realloc<T>(&data, capacity);
        }

        MemUtils::memcpy<T>(data, other->data, other->size);
        size = other->size;
    }

    void resize(int count)
    {
        if (capacity < count)
        {
            capacity = count * 2;
            MemUtils::realloc<T>(&data, capacity);
        }

        size = count;
    }

    void resize_by_one()
    {
        resize(size + 1);
    }

    inline T& back()
    {
        assert(size > 0);

        return *(data + (size - 1));
    }

    void push_back(T t)
    {
        if (size == capacity)
        {
            capacity = capacity * 2;
            MemUtils::realloc<T>(&data, capacity);
        }

        *(data + size) = t;
        size += 1;
    }

    inline void pop_back()
    {
        assert(size > 0);

        size -= 1;
    }

    void clear()
    {
        size = 0;
    }
};
