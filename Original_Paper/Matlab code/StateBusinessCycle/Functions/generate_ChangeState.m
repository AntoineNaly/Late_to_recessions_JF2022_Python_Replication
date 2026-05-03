function changeState = generate_ChangeState(S_T,states)

    s = S_T;


    n = length(s);
    m = length(states);

    changeState = zeros(m);

     for t=2:n

        st1 = s(t-1);
        st  = s(t);

        changeState(st1,st) = changeState(st1,st)+1;

     end
end