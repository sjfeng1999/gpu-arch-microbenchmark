import os
import argparse


def camel_to_snake(name):
    token_list = name.split("_")
    camel_name = ""
    for i, token in enumerate(token_list):
        if i == 0:
            camel_name += token 
        else:
            camel_name += token.capitalize()
    return camel_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-arch", type=int, default=75)

    args = parser.parse_args()

    ARCH_LIST = [70, 75, 80]
    KERNEL_LIST = ["memory_latency", 
                   "cache_linesize", 
                   "reg_with_bankconflict", "reg_without_bankconflict"]

    if args.arch not in ARCH_LIST:
        print("Unsupported Gpu Arch: ", args.arch)
        exit()

    print(">>>")
    for kernel in KERNEL_LIST:
        source_sass = f"{kernel}.sass"
        target_cubin = f"{kernel}.cubin"
        target_kernel = camel_to_snake(kernel)
        compile_command = f"python3 -m turingas.main -i ../sass_cubin/{source_sass} -o ../sass_cubin/{target_cubin} -arch {args.arch} -name {target_kernel}"

        print(f"    compile kernel: {target_kernel}")
        os.system(compile_command)
    print("<<<")