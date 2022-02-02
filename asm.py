s = '''     cij = _mm512_loadu_pd(C + (i + j * lda));
            cij1 = _mm512_loadu_pd(C + (i + (j+1) * lda));
            cij2 = _mm512_loadu_pd(C + (i + (j+2) * lda));
            cij3 = _mm512_loadu_pd(C + (i + (j+3) * lda));
            cij4 = _mm512_loadu_pd(C + (i + (j+4) * lda));
            cij5 = _mm512_loadu_pd(C + (i + (j+5) * lda));
            cij6 = _mm512_loadu_pd(C + (i + (j+6) * lda));
            cij7 = _mm512_loadu_pd(C + (i + (j+7) * lda));

            ci8j = _mm512_loadu_pd(C + (i + 8 + j * lda));
            ci8j1 = _mm512_loadu_pd(C + (i + 8 + (j+1) * lda));
            ci8j2 = _mm512_loadu_pd(C + (i + 8 + (j+2) * lda));
            ci8j3 = _mm512_loadu_pd(C + (i + 8 + (j+3) * lda));
            ci8j4 = _mm512_loadu_pd(C + (i + 8 + (j+4) * lda));
            ci8j5 = _mm512_loadu_pd(C + (i + 8 + (j+5) * lda));
            ci8j6 = _mm512_loadu_pd(C + (i + 8 + (j+6) * lda));
            ci8j7 = _mm512_loadu_pd(C + (i + 8 + (j+7) * lda));
'''

s2 = '''         aik = _mm512_loadu_pd(A + (i + k * lda));
                ai8k = _mm512_loadu_pd(A + (i + 8 + k * lda));

                bkj = _mm512_set1_pd(B[k + j * lda]);
                cij = _mm512_fmadd_pd(aik, bkj, cij);
                ci8j = _mm512_fmadd_pd(ai8k, bkj, ci8j);

                bkj = _mm512_set1_pd(B[k + (j+1) * lda]);
                cij1 = _mm512_fmadd_pd(aik, bkj, cij1);
                ci8j1 = _mm512_fmadd_pd(ai8k, bkj, ci8j1);

                bkj = _mm512_set1_pd(B[k + (j+2) * lda]);
                cij2 = _mm512_fmadd_pd(aik, bkj, cij2);
                ci8j2 = _mm512_fmadd_pd(ai8k, bkj, ci8j2);

                bkj = _mm512_set1_pd(B[k + (j+3) * lda]);
                cij3 = _mm512_fmadd_pd(aik, bkj, cij3);
                ci8j3 = _mm512_fmadd_pd(ai8k, bkj, ci8j3);

                bkj = _mm512_set1_pd(B[k + (j+4) * lda]);
                cij4 = _mm512_fmadd_pd(aik, bkj, cij4);
                ci8j4 = _mm512_fmadd_pd(ai8k, bkj, ci8j4);

                bkj = _mm512_set1_pd(B[k + (j+5) * lda]);
                cij5 = _mm512_fmadd_pd(aik, bkj, cij5);
                ci8j5 = _mm512_fmadd_pd(ai8k, bkj, ci8j5);

                bkj = _mm512_set1_pd(B[k + (j+6) * lda]);
                cij6 = _mm512_fmadd_pd(aik, bkj, cij6);
                ci8j6 = _mm512_fmadd_pd(ai8k, bkj, ci8j6);

                bkj = _mm512_set1_pd(B[k + (j+7) * lda]);
                cij7 = _mm512_fmadd_pd(aik, bkj, cij7);
                ci8j7 = _mm512_fmadd_pd(ai8k, bkj, ci8j7);
'''

s3 = '''
            _mm512_store_pd(C + i + j * lda, cij);
            _mm512_store_pd(C + i + (j+1) * lda, cij1);
            _mm512_store_pd(C + i + (j+2) * lda, cij2);
            _mm512_store_pd(C + i + (j+3) * lda, cij3);
            _mm512_store_pd(C + i + (j+4) * lda, cij4);
            _mm512_store_pd(C + i + (j+5) * lda, cij5);
            _mm512_store_pd(C + i + (j+6) * lda, cij6);
            _mm512_store_pd(C + i + (j+7) * lda, cij7);


            _mm512_store_pd(C + i + 8 + j * lda, ci8j);
            _mm512_store_pd(C + i + 8 + (j+1) * lda, ci8j1);
            _mm512_store_pd(C + i + 8 + (j+2) * lda, ci8j2);
            _mm512_store_pd(C + i + 8 + (j+3) * lda, ci8j3);
            _mm512_store_pd(C + i + 8 + (j+4) * lda, ci8j4);
            _mm512_store_pd(C + i + 8 + (j+5) * lda, ci8j5);
            _mm512_store_pd(C + i + 8 + (j+6) * lda, ci8j6);
            _mm512_store_pd(C + i + 8 + (j+7) * lda, ci8j7);
'''

# Assembly operands cannot be arrays with indices, this converts them to a consistent variable name.
def array_and_index_to_variable(value):
    val = ""
    for char in value:
        if char == "i" or char=="j" or char=="k" or char=="C" or char=="A" or char=="B":
            val += char
        else:
            try:
                int(char)
                val += str(char)
            except:
                pass
    return val


def parse_load(line):
    register_to_load_into = line[:line.find(" =")]
    value_line = line[line.find("_mm512_loadu_pd(")+len("_mm512_loadu_pd("):]
    value_to_add = value_line[:value_line.rfind(")")]
    value_variable = array_and_index_to_variable(value_to_add.replace(" ",""))
    asm_line = "\"vmovapd (%[{}]), %[{}]\\n\\t\"".format(value_variable, register_to_load_into)
    assignment_line = "double* {} = {};".format(value_variable, value_to_add)
    return asm_line, value_variable, register_to_load_into, assignment_line

# Takes in a set and two adds and terms them into two embedded broadcast adds.
def parse_set_and_adds(lines):
    reg_to_broadcast_line = lines[0][lines[0].find("_mm512_set1_pd(")+len("_mm512_set1_pd("):]
    reg_to_broadcast = reg_to_broadcast_line[:reg_to_broadcast_line.rfind(")")].replace(" ", "")
    
    def gen_vfmadd_line(registers):
        return "\"vfmadd231pd %[{}], %[{}], %([{}])\"".format(registers_to_add[1], registers_to_add[0], registers_to_add[2])

    
    registers_to_add_line = lines[1][lines[1].find("_mm512_fmadd_pd(")+len("_mm512_fmadd_pd("):]
    registers_to_add = registers_to_add_line[:registers_to_add_line.rfind(")")].split(", ")
    line_one = gen_vfmadd_line(registers_to_add)

    registers_to_add_line = lines[2][lines[2].find("_mm512_fmadd_pd(")+len("_mm512_fmadd_pd("):]
    registers_to_add = registers_to_add_line[:registers_to_add_line.rfind(")")].split(", ")
    line_two = gen_vfmadd_line(registers_to_add)
    return line_one, line_two, registers_to_add

def parse_store(line):
    value_line = line[line.find("_mm512_store_pd(")+len("_mm512_store_pd("):]
    values = value_line[:value_line.rfind(")")]
    values = values.split(", ")
    memory_location = values[0].replace(" ", "")
    register = values[1].replace(" ", "")
    memory_location = array_and_index_to_variable(memory_location)

    asm_line = "\"vmovapd (%[{}]), %[{}]\"".format(register, memory_location)
    return asm_line, memory_location, register

def parse_string(s):
    asm_string = "asm volatile (\n"
    input_registers = []
    output_registers = []
    lines = s.split("\n")
    i = 0

    variable_to_assignment_line = {}
    while i < len(lines):
        line = lines[i]
        line = line.strip()
        line = line.lstrip()
        if "load" in line:
            asm_line, input_register, output_register, assignment_line = parse_load(line)
            variable_to_assignment_line[input_register] = assignment_line
            if not input_register in input_registers:
                input_registers.append(input_register)
            if not output_register in output_registers:
                output_registers.append(output_register)
            asm_string += "\t" + asm_line + "\n"
        elif "set" in line:
            # Process as a set of 3 instructions.
            add_lines = [line, lines[i+1].strip().lstrip(), lines[i+2].strip().lstrip()]
            i += 2
            asm_line_one, asm_line_two, registers = parse_set_and_adds(add_lines)
            asm_string += "\t" + asm_line_one + "\n"
            asm_string += "\t" + asm_line_two + "\n"
            for register in registers:
                if not register in input_registers:
                    input_registers.append(register)
        elif "store" in line:
            asm_line, input_register, output_register = parse_store(line)
            if not input_register in input_registers:
                input_registers.append(input_register)
            if not output_register in output_registers:
                output_registers.append(output_register)
            asm_string += "\t" + asm_line + "\n"
        i += 1
    if len(output_registers) > 0:
        asm_string += "\t: "
        for i, register in enumerate(output_registers):
            if i == 0:
                asm_string += "[{}] \"+v\" ({})".format(register, register)
            else:
                asm_string += ",\n\t  [{}] \"+v\" ({})".format(register, register)
    if len(input_registers) > 0:
        asm_string += "\n\t: "
        for i, register in enumerate(input_registers):
            if i == 0:
                asm_string += "[{}] \"r\" ({})".format(register, register)
            else:
                asm_string += ",\n\t  [{}] \"r\" ({})".format(register, register)
    asm_string += "\n\t: \"memory\"\n"
    asm_string += ");"

    for line in variable_to_assignment_line.values():
        print(line)
    print(asm_string)   

parse_string(s)
parse_string(s2)
parse_string(s3)