def scale_to_range(labels, _min, _max):
    return list(map(lambda x: (x - _min) / (_max - _min), labels))


def get_preprocessing_function(
        tokenizer,
        sentence1_key,
        sentence2_key,
        condition_key,
        similarity_key,
        padding,
        max_seq_length,
        model_args,
        scale=None,
        condition_only=False,
        sentences_only=False,
        ):
    'Returns a the preprocessing function for each encoding type'
    if model_args.encoding_type == 'bi_encoder':
        if condition_only or sentences_only:
            raise ValueError('condition_only and sentences_only doesn\'t apply to bi_encoder')
        def preprocess_function(examples):
            sep_token_id = tokenizer.convert_tokens_to_ids("</s>")
            '''sent1_args = (examples[sentence1_key], examples[condition_key])
            sent1_result = tokenizer(*sent1_args, padding=padding, max_length=max_seq_length, truncation=True)
            sent2_args = (examples[sentence2_key], examples[condition_key])
            sent2_result = tokenizer(*sent2_args, padding=padding, max_length=max_seq_length, truncation=True)
            sent1_result['input_ids_2'] = sent2_result['input_ids']
            sent1_result['attention_mask_2'] = sent2_result['attention_mask']

            if 'token_type_ids' in sent2_result:
                sent1_result['token_type_ids_2'] = sent2_result['token_type_ids']'''

            sent1_args = (examples[sentence1_key], )
            sent1_result = tokenizer(*sent1_args, padding=padding, max_length=max_seq_length, truncation=True)
            sent2_args = (examples[sentence2_key], )
            sent2_result = tokenizer(*sent2_args, padding=padding, max_length=max_seq_length, truncation=True)
            sent3_args = (examples[condition_key], )
            sent3_result = tokenizer(*sent3_args, padding=padding, max_length=max_seq_length, truncation=True)
            
            '''sent1_result['input_ids'] = [input_ids + [sep_token_id] for input_ids in sent1_result['input_ids']]
            sent1_result['attention_mask'] = [attention_mask + [1] for attention_mask in sent1_result['attention_mask']]

            sent1_result['input_ids_2'] = [input_ids + [sep_token_id] for input_ids in sent2_result['input_ids']]
            sent1_result['attention_mask_2'] = [attention_mask + [1] for attention_mask in sent2_result['attention_mask']]
            sent1_result['input_ids_3'] = sent3_result['input_ids']
            sent1_result['attention_mask_3'] = sent3_result['attention_mask']'''

            sent1_result['input_ids_2'] = sent2_result['input_ids']
            sent1_result['attention_mask_2'] = sent2_result['attention_mask']
            sent1_result['input_ids_3'] = [[sep_token_id] + input_ids for input_ids in sent3_result['input_ids']]
            sent1_result['attention_mask_3'] = [[1] + attention_mask for attention_mask in sent3_result['attention_mask']]

            if 'token_type_ids' in sent2_result:
                sent1_result['token_type_ids_2'] = sent2_result['token_type_ids']
                sent1_result['token_type_ids_3'] = sent3_result['token_type_ids']

            sent1_result['labels'] = examples[similarity_key]
            if scale is not None:
                _min, _max = scale
                for label in sent1_result['labels']:
                    if (label < _min or label > _max) and label != -1:
                        raise ValueError(f'Label {label} is not in the range [{_min}, {_max}]')
                sent1_result['labels'] = scale_to_range(sent1_result['labels'], _min, _max)
            return sent1_result
    elif model_args.encoding_type == 'cross_encoder':
        def preprocess_function(examples):
            if condition_only:
                input_args = examples[condition_key]
            elif sentences_only:
                input_args = list(map(lambda x: ' '.join([x[0], tokenizer.sep_token, x[1]]), zip(examples[sentence1_key], examples[sentence2_key])))
            else:
                input_args = list(map(lambda x: ' '.join([x[0], tokenizer.sep_token, x[1], tokenizer.sep_token, x[2]]), zip(examples[sentence1_key], examples[sentence2_key], examples[condition_key])))
            result = tokenizer(input_args, padding=padding, max_length=max_seq_length, truncation=True)
            result['labels'] = examples[similarity_key]
            if scale is not None:
                _min, _max = scale
                for label in result['labels']:
                    if (label < _min or label > _max) and label != -1:
                        raise ValueError(f'Label {label} is not in the range [{_min}, {_max}]')
                result['labels'] = scale_to_range(result['labels'], _min, _max)
            return result
    elif model_args.encoding_type == 'tri_encoder':
        if condition_only or sentences_only:
            raise ValueError('condition_only and sentences_only doesn\'t apply to tri_encoder')
        def preprocess_function(examples):
            sent1_args = (examples[sentence1_key], )
            sent1_result = tokenizer(*sent1_args, padding=padding, max_length=max_seq_length, truncation=True)
            sent2_args = (examples[sentence2_key], )
            sent2_result = tokenizer(*sent2_args, padding=padding, max_length=max_seq_length, truncation=True)
            sent3_args = (examples[condition_key], )
            sent3_result = tokenizer(*sent3_args, padding=padding, max_length=max_seq_length, truncation=True)
            sent1_result['input_ids_2'] = sent2_result['input_ids']
            sent1_result['attention_mask_2'] = sent2_result['attention_mask']
            sent1_result['input_ids_3'] = sent3_result['input_ids']
            sent1_result['attention_mask_3'] = sent3_result['attention_mask']
            if 'token_type_ids' in sent2_result:
                sent1_result['token_type_ids_2'] = sent2_result['token_type_ids']
                sent1_result['token_type_ids_3'] = sent3_result['token_type_ids']
            sent1_result['labels'] = examples[similarity_key]
            if scale is not None:
                _min, _max = scale
                for label in sent1_result['labels']:
                    if (label < _min or label > _max) and label != -1:
                        raise ValueError(f'Label {label} is not in the range [{_min}, {_max}]')
                sent1_result['labels'] = scale_to_range(sent1_result['labels'], _min, _max)
            return sent1_result
    else:
        raise ValueError(f'Invalid model type: {model_args.encoding_type}')
    return preprocess_function


def get_add_supervision_function_(
        tokenizer, 
        sentence1_key,
        sentence2_key,
        condition_key,
        sentence1_keyword,
        sentence2_keyword,
        condition_keyword,
        ):
    def get_add_supervision_function(file_name):
        import json
        with open(file_name, 'r') as file:
            dataset = json.load(file)
        def add_supervision_function(example, index): 
            try:
                data = dataset[str(index)]
                assert data[condition_key] == example[condition_key] and data[sentence1_key] == example[sentence1_key]
                supervision = {
                    "condition_key":data[condition_keyword],
                    "sentence1_key":data[sentence1_keyword],
                    "sentence2_key":data[sentence2_keyword],
                    "key_ids": get_idx(data[sentence1_keyword], tokenizer.convert_ids_to_tokens(example["input_ids"])),
                    "key_ids_2": get_idx(data[sentence2_keyword], tokenizer.convert_ids_to_tokens(example["input_ids_2"])),
                    "key_ids_3": get_idx(data[condition_keyword], tokenizer.convert_ids_to_tokens(example["input_ids_3"])),
                }
            except:
                supervision = {
                    "condition_key": '[None]',
                    "sentence1_key": '[None]',
                    "sentence2_key": '[None]',
                    "key_ids": [0] * len(example["input_ids"]),
                    "key_ids_2": [0] * len(example["input_ids_2"]),
                    "key_ids_3": [0] * len(example["input_ids_3"]),
                }
            return supervision
        return add_supervision_function
    return get_add_supervision_function


def get_idx(key_words, texts):
    length = len(texts)
    key_words = key_words.split(",")
    texts = [text.replace('Ġ', '') for text in texts]
    result = []
    
    for key_word in key_words:
        index = [idx for idx,  text in enumerate(texts) if text in key_word]
        index = find_longest_consecutive_sequence(index)
        result += index
    return [int(i in result) for i in range(length)]

def find_longest_consecutive_sequence(nums):
    nums_set = set(nums)  # 将列表转换为集合，以便快速检查数字是否存在
    max_length = 0
    result = []

    for num in nums:
        if num - 1 not in nums_set:  # 只处理序列的起始数字
            current_length = 1
            current_num = num + 1

            # 继续查找连续的数字
            while current_num in nums_set:
                current_length += 1
                current_num += 1

            # 更新最长序列的长度和结果
            if current_length > max_length:
                max_length = current_length
                result = list(range(num, num + current_length))

    return result