int helper(int x) {
    return x;
}
void caller(int x) {
    char buf[8];
    int n = helper(x);
    buf[n] = 0;
}
