function [entropies] = find_all_nsb_entropies(in_filename, out_filename, bins_n, variables_n)
    load(in_filename, "disc_data")
    qfun=1;
    precision=.1;
    
    v_indeces = (1:variables_n);
    distr_cards = zeros(1, variables_n, 'uint8') + bins_n;

    disc_data_table = zeros(distr_cards, 'double');  % 'double' because of nsb errors
    for i = 1:size(disc_data, 1)
        row = num2cell(disc_data(i,:));
        index = sub2ind(size(disc_data_table), row{:});
        disc_data_table(index) = disc_data_table(index) + 1; 
    end
    
    entropies = struct();
    mem = struct();
    for i = (1:(2^variables_n)-1)
        %Convert i into binary, convert each digit in binary to a boolean
        %and store that array of booleans
        indices = logical(bitget(i,(1:variables_n)));
        %Use the array of booleans to extract the members of the original
        %set, and store the set containing these members in the powerset
        choosen_variables = v_indeces(indices);
        excluded_varibales = v_indeces(~indices);
        choosen_variables_str = num2str(reshape(choosen_variables, 1, []));
        
        if length(choosen_variables) == variables_n
            tmp_data_table = disc_data_table;
        else
            % in matlab just sum(disc_data_table, excluded_varibales, 'native');
            tmp_data_table = disc_data_table;
            for j = length(excluded_varibales):-1:1
                tmp_data_table = sum(tmp_data_table, excluded_varibales(j));
            end
            
        end
        serialized_data = reshape(tmp_data_table, 1, []);
        serialized_data_str = num2str(reshape(serialized_data, 1, []));

        if not(isfield(mem, serialized_data_str))
            K = length(serialized_data);
            
            nx = serialized_data(serialized_data>0);
            kx = ones(size(nx), 'double');

            entropy = find_nsb_entropy(kx, nx, K, precision,qfun);
            mem.(serialized_data_str) = entropy;
        end
        entropies.(choosen_variables_str) = getfield(mem, serialized_data_str);  
    end
    save("-v7", out_filename, "entropies")
end