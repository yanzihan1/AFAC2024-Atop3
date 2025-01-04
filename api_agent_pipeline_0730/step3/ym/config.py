### 需要的参数都在这里配置

class config(object):
    def __init__(self):
        self.train_xlsx = '/mnt/mcu/yanzihan/agent-code/data/data-0520/标准名.xlsx'  # read
        self.train_data_xlsx = '/mnt/mcu/yanzihan/agent-code/data/data-0520/train.xlsx'  # read
        self.dev_data_xlsx = '/mnt/mcu/yanzihan/agent-code/data/data-0520/dev.xlsx'  # read
        self.test_b_data_xlsx = '/mnt/mcu/yanzihan/agent-code/data/b_data/test_b_without_label.xlsx'  # read

        self.api_path = '/mnt/mcu/yanzihan/agent-code/data/data-0520/api_定义.json'  # read  -
        self.train_json = '/mnt/mcu/yanzihan/agent-code/ChatGLM2-6B/ptuning/data/json_data/train.json'  # 官方脚本跑出来的train.json # read
        self.dev_json = '/mnt/mcu/yanzihan/agent-code/ChatGLM2-6B/ptuning/data/json_data/dev.json'  # 官方脚本跑出来的dev.json # read
        self.qwen2_model = '/mnt/nas_new/plms/llm3/' # step3训练好的 qwen2模型的地址

        #### 只有上面这里的数据是需要读的 需要事先写好读取的路径

        # keywords 相关的文件
        self.train_keywords = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/step1/train_keywords.json"
        self.dev_keywords = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/step1/dev_keywords.json"
        self.test_keywords = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/step1/test_keywords.json"
        self.recall_keywords_reranker_model_path = "/mnt/nas/yanzihan/reranker_recall/"



        ### step 1
        self.train_recall_api_keywords = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/step1/train_recall_api_keywords.json"  # reranker训练召回关键词 # write
        self.dev_recall_api_keywords = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/step1/dev_recall_api_keywords.json"  # 验证召回关键词 # write
        self.neg_topK = 3  # 训练bge_reranker的负样本采样 默认是1:3

        ### step 2
        self.train_match_api = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/step2/train_match_api.json"  # reranker训练召回 可能需要的api # write
        self.dev_match_api = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/step2/dev_match_api.json"  # reranker训练召回 可能需要的api # write
        self.api_neg_topK = 12  # 训练bge_reranker的负样本采样

        ### step3
        self.recall_api_reranker_model_path = "/mnt/nas_new/yanzihan/reranker/"  # step2 训练得到的reranker模型地址  -
        # self.recall_api_reranker_model_path = "/mnt/nas_new/yanzihan/reranker_recall_api_v2/"
        self.recall_api_topk = 20  # step3中训练的时候需要召回topk个可能需要用到的api
        self.step3_save_path = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/step3/step3_LLM_train.json"  # step3 训练文件 -
        self.step3_dev_save_path = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/step3/step3_LLM_dev.json"  # step3 验证文件 -

        ### step4
        self.step4_save_path = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/step4/step4_LLM_train.json"  # step4 训练文件
        self.step4_dev_save_path = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/step4/step4_LLM_dev.json"  # step4 dev输入文件
        self.step4_dev_output_path = "/mnt/mcu/yanzihan/tob_baichuan/aftc/step4_test/submitQWEN2_0621.json"  # step4 dev输入文件

        self.recall_keywords_topk = 15

        ### Test Config
        # 这个是step3在测试集上最终选择的api 参考格式 api_top20/step3_v2.json
        self.test_step3_save_path = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/test_piepline/step3_LLM_test.json"  # step3 测试文件  -

        # 这个是step3在测试集上最终选择的api 参考格式 api_top20/step3_v2.json
        self.test_step3_final_choose_path = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/test_piepline/step3_res.json"  # step3 测试文件  -

        # self.test_b_step3_save_path = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/test_piepline/b_test/step3/step3_b_LLM_test0725.json"  # step3 测试文件

        self.test_step3_save_path_v2 = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/test_piepline/step3_LLM_test_v2.json"  # step3 测试文件

        self.step3_LLM_res = "/mnt/mcu/chongyangwang/code/llama/ant/step3/test_step3_query_to_qwen2.json"

        # self.step3_LLM_res = "/mnt/mcu/yanzihan/tob_baichuan/aftc/step3_dev/step3_dev_query_to_choose.json"
        self.step3_dev_LLM_res = "/mnt/mcu/yanzihan/tob_baichuan/aftc/step3_dev/step3_dev_query_to_choose.json"

        # self.test_step4_save_path = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/test_piepline/step4_LLM_test.json"  # step4 测试文件
        self.test_step4_save_path = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/test_piepline/b_test/step4_b_LLM_test.json"  # step4 测试文件  -

        self.step4_LLM_res = "/mnt/mcu/yanzihan/agent-code/data/do_train_data/test_piepline/submit.json"
