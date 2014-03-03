#pragma once

namespace ml {

template <typename T>
int sign(const T& t) {
    if (t > 0)
        return 1;
    if (t < 0)
        return -1;
    return 0;
}

} // namespace ml

