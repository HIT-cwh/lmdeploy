# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
import os
import os.path as osp
import random

os.environ['TM_LOG_LEVEL'] = 'ERROR'


@dataclasses.dataclass
class GenParam:
    top_p: float
    top_k: float
    temperature: float
    repetition_penalty: float
    sequence_start: bool = False
    sequence_end: bool = False
    step: int = 0
    request_output_len: int = 512


def input_prompt(model_name):
    """Input a prompt in the consolo interface."""
    if model_name == 'codellama':
        print('\nenter !! to end the input >>>\n', end='')
        sentinel = '!!'
    else:
        print('\ndouble enter to end input >>> ', end='')
        sentinel = ''  # ends when this string is seen
    return '\n'.join(iter(input, sentinel))


def valid_str(string, coding='utf-8'):
    """decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret


def get_gen_param(cap,
                  sampling_param,
                  nth_round,
                  step,
                  request_output_len=512,
                  **kwargs):
    """return parameters used by token generation."""
    gen_param = GenParam(**dataclasses.asdict(sampling_param),
                         request_output_len=request_output_len)
    # Fix me later. turbomind.py doesn't support None top_k
    if gen_param.top_k is None:
        gen_param.top_k = 40

    if cap == 'chat':
        gen_param.sequence_start = (nth_round == 1)
        gen_param.sequence_end = False
        gen_param.step = step
    else:
        gen_param.sequence_start = True
        gen_param.sequence_end = True
        gen_param.step = 0
    return gen_param


def main(model_path,
         session_id: int = 1,
         cap: str = 'chat',
         tp: int = 1,
         stream_output: bool = True,
         request_output_len: int = 512,
         **kwargs):
    """An example to perform model inference through the command line
    interface.

    Args:
        model_path (str): the path of the deployed model
        session_id (int): the identical id of a session
        cap (str): the capability of a model. For example, codellama has
            the ability among ['completion', 'infilling', 'chat', 'python']
        tp (int): GPU number used in tensor parallelism
        stream_output (bool): indicator for streaming output or not
        **kwarg (dict): other arguments for initializing model's chat template
    """
    from lmdeploy import turbomind as tm
    from lmdeploy.tokenizer import Tokenizer

    tokenizer_model_path = osp.join(model_path, 'triton_models', 'tokenizer')
    tokenizer = Tokenizer(tokenizer_model_path)
    tm_model = tm.TurboMind(model_path,
                            eos_id=tokenizer.eos_token_id,
                            tp=tp,
                            capability=cap,
                            **kwargs)
    generator = tm_model.create_instance()

    nth_round = 1
    step = 0
    seed = random.getrandbits(64)
    model_name = tm_model.model_name
    model = tm_model.model

    print(f'session {session_id}')
    while True:
        prompt = input_prompt(model_name)
        if prompt == 'exit':
            exit(0)
        elif prompt == 'end':
            prompt = model.get_prompt('', nth_round == 1)
            input_ids = tokenizer.encode(prompt)
            for outputs in generator.stream_infer(
                    session_id=session_id,
                    input_ids=[input_ids],
                    request_output_len=request_output_len,
                    sequence_start=False,
                    sequence_end=True,
                    stream_output=stream_output):
                pass
            nth_round = 1
            step = 0
            seed = random.getrandbits(64)
        else:
            prompt = model.get_prompt(prompt, nth_round == 1)
            input_ids = tokenizer.encode(prompt)
            if step + len(
                    input_ids) + request_output_len >= tm_model.session_len:
                print('WARNING: exceed session max length.'
                      ' Please end the session.')
                continue

            gen_param = get_gen_param(cap, model.sampling_param, nth_round,
                                      step, request_output_len, **kwargs)

            print(f'{prompt} ', end='', flush=True)
            response_size = 0
            for outputs in generator.stream_infer(
                    session_id=session_id,
                    input_ids=[input_ids],
                    stream_output=stream_output,
                    **dataclasses.asdict(gen_param),
                    ignore_eos=False,
                    random_seed=seed if nth_round == 1 else None):
                res, tokens = outputs[0]
                # decode res
                response = tokenizer.decode(res.tolist(), offset=response_size)
                # utf-8 char at the end means it's a potential unfinished
                # byte sequence, continue to concate it with the next
                # sequence and decode them together
                if response.endswith('�'):
                    continue
                response = valid_str(response)
                print(f'{response}', end='', flush=True)
                response_size = tokens

            # update step
            step += len(input_ids) + tokens
            print()

            nth_round += 1


if __name__ == '__main__':
    import fire

    fire.Fire(main)
