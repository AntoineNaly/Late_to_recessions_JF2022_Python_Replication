  function s = bingen(p0,p1, m)

        pr0 = p0/(p0+p1);  
        u = rand(m);
        s =  1 - (u < pr0);

    end