from transformers import AutoTokenizer

tokenizer_path = "/data/lhc/models_new/Qwen3-0.6B_sleep_st_44_100hz_eeg7.5s-step7.5s_emo2.0s-step0.25s_win_all_tokenizer_qwen_tok9689_bal0.5_sqrt_inverse_202505130318_train/final_20250514_052715/"
token_id_to_check = 151669

try:
    print(f"正在从路径加载 Tokenizer: {tokenizer_path}")
    # 确保 trust_remote_code=True，因为 Qwen 模型通常需要它
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print("Tokenizer 加载成功。")

    vocab_size = tokenizer.vocab_size
    len_tokenizer = len(tokenizer) # 对于添加了token的情况，len(tokenizer) 通常更准确

    print(f"tokenizer.vocab_size: {vocab_size}")
    print(f"len(tokenizer): {len_tokenizer}") # 这通常也反映了词汇表大小，特别是添加了新token后

    print(f"\n所有特殊 Token: {tokenizer.all_special_tokens}")
    print(f"所有特殊 Token ID: {tokenizer.all_special_ids}")

    # 检查 token_id_to_check 是否在词汇表范围内
    # 通常，如果 token ID >= vocab_size (或 len(tokenizer))，则它超出了范围
    is_in_vocab_explicit_check = token_id_to_check < vocab_size
    is_in_vocab_len_check = token_id_to_check < len_tokenizer


    print(f"\n检查 Token ID {token_id_to_check}:")
    print(f"  是否 < tokenizer.vocab_size ({vocab_size})? {is_in_vocab_explicit_check}")
    print(f"  是否 < len(tokenizer) ({len_tokenizer})? {is_in_vocab_len_check}")

    effective_vocab_size = len_tokenizer # 优先使用 len(tokenizer)
    if effective_vocab_size <= token_id_to_check :
        print(f"  错误: Token ID {token_id_to_check} >= 有效词汇表大小 ({effective_vocab_size})。该 Token ID 超出范围。")
    else:
        print(f"  Token ID {token_id_to_check} 在有效词汇表大小 ({effective_vocab_size}) 范围内。")
        try:
            decoded_token = tokenizer.decode([token_id_to_check])
            print(f"  成功解码 Token ID {token_id_to_check} 为: '{decoded_token}'")
            
            # 进一步检查，尝试直接转换 ID 到 token 字符串
            # token_str = tokenizer.convert_ids_to_tokens([token_id_to_check])
            # print(f"  tokenizer.convert_ids_to_tokens([{token_id_to_check}]): {token_str}")
        except Exception as e_decode:
            print(f"  解码 Token ID {token_id_to_check} 时出错: {e_decode}")
            # 即使在词汇表大小范围内，某些ID也可能无法直接解码，特别是如果它们是内部的或未使用的ID

    # 尝试查看 tokenizer 配置文件中的 vocab_size (如果可用)
    if hasattr(tokenizer, 'vocab') and tokenizer.vocab and token_id_to_check in tokenizer.vocab:
        print(f"  Token ID {token_id_to_check} 在 tokenizer.vocab 中找到: {tokenizer.vocab[token_id_to_check]}")
    
    # 打印一些可能相关的ID，比如eos_token_id
    if tokenizer.eos_token_id is not None:
        print(f"\nEOS token ID: {tokenizer.eos_token_id}, Token: '{tokenizer.decode([tokenizer.eos_token_id])}'")
    if tokenizer.bos_token_id is not None:
        print(f"BOS token ID: {tokenizer.bos_token_id}, Token: '{tokenizer.decode([tokenizer.bos_token_id])}'")
    if tokenizer.pad_token_id is not None:
        print(f"PAD token ID: {tokenizer.pad_token_id}, Token: '{tokenizer.decode([tokenizer.pad_token_id])}'")

except FileNotFoundError:
    print(f"错误: Tokenizer 路径未找到: {tokenizer_path}")
except Exception as e:
    print(f"加载或检查 Tokenizer 时发生错误: {e}")
    import traceback
    traceback.print_exc()
