function freq = fftfreq(n, dt)
    if mod(n, 2) == 0
        freq = (0:n/2) / (n * dt);
        freq = [freq, -fliplr(freq(2:end-1))];
    else
        freq = (0:(n-1)/2) / (n * dt);
        freq = [freq, -fliplr(freq)];
    end
end