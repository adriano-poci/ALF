/'cz out of prior bounds, setting to 0.0'/ {
    p;n;
    /velz = 0.0/ {
        s/velz = 0.0/velz = 999/;
        p;d;
            }
}
p;
