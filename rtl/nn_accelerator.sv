module nn_accelerator #(
    parameter int INPUT_SIZE    = 64,
    parameter int HIDDEN_SIZE   = 16,
    parameter int OUTPUT_SIZE   = 10,
    parameter int INPUT_WIDTH   = 8,
    parameter int WEIGHT_WIDTH  = 8,
    parameter int ACC_WIDTH     = 32,
    parameter int SCALE_BITS    = 7
)(
    input  logic clk,
    input  logic rst,
    input  logic start,

    output logic done,
    output logic [3:0] prediction
);

    logic signed [INPUT_WIDTH-1:0]  input_mem  [0:INPUT_SIZE-1];

    logic signed [WEIGHT_WIDTH-1:0] weights_l1 [0:INPUT_SIZE*HIDDEN_SIZE-1];
    logic signed [ACC_WIDTH-1:0]    biases_l1  [0:HIDDEN_SIZE-1];

    logic signed [WEIGHT_WIDTH-1:0] weights_l2 [0:HIDDEN_SIZE*OUTPUT_SIZE-1];
    logic signed [ACC_WIDTH-1:0]    biases_l2  [0:OUTPUT_SIZE-1];

    logic signed [ACC_WIDTH-1:0] hidden_mem [0:HIDDEN_SIZE-1];
    logic signed [ACC_WIDTH-1:0] output_mem [0:OUTPUT_SIZE-1];

    initial begin
        // $readmemh("data/mem/input.mem",      input_mem);
        $readmemh("data/mem/weights_l1.mem", weights_l1);
        $readmemh("data/mem/biases_l1.mem",  biases_l1);
        $readmemh("data/mem/weights_l2.mem", weights_l2);
        $readmemh("data/mem/biases_l2.mem",  biases_l2);
    end

    typedef enum logic [3:0] {
        IDLE,
        L1_INIT,
        L1_MAC,
        L1_STORE,
        L2_INIT,
        L2_MAC,
        L2_STORE,
        ARGMAX_INIT,
        ARGMAX_RUN,
        ARGMAX_DONE,
        DONE_STATE
    } state_t;

    state_t state;

    logic [$clog2(INPUT_SIZE)-1:0]  input_idx;
    logic [$clog2(HIDDEN_SIZE)-1:0] hidden_idx;
    logic [$clog2(OUTPUT_SIZE)-1:0] output_idx;
    logic [$clog2(HIDDEN_SIZE)-1:0] l2_hidden_idx;
    logic [$clog2(OUTPUT_SIZE)-1:0] argmax_idx;

    logic signed [ACC_WIDTH-1:0] acc_reg;

    logic signed [ACC_WIDTH-1:0]    mac_input;
    logic signed [WEIGHT_WIDTH-1:0] mac_weight;
    logic signed [ACC_WIDTH-1:0]    mac_acc_out;

    logic signed [ACC_WIDTH-1:0] max_value;
    logic [3:0] max_index;

    mac_unit #(
        .DATA_WIDTH(ACC_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) mac_inst (
        .input_value(mac_input),
        .weight_value(mac_weight),
        .acc_in(acc_reg),
        .acc_out(mac_acc_out)
    );

    always_comb begin
        mac_input = '0;
        mac_weight = '0;

        case (state)
            L1_MAC: begin
                mac_input = $signed(input_mem[input_idx]);
                mac_weight = weights_l1[hidden_idx * INPUT_SIZE + input_idx];
            end

            L2_MAC: begin
                mac_input = hidden_mem[l2_hidden_idx];
                mac_weight = weights_l2[output_idx * HIDDEN_SIZE + l2_hidden_idx];
            end

            default: begin
                mac_input = '0;
                mac_weight = '0;
            end
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            done <= 1'b0;
            prediction <= 4'd0;

            input_idx <= '0;
            hidden_idx <= '0;
            output_idx <= '0;
            l2_hidden_idx <= '0;
            argmax_idx <= '0;

            acc_reg <= '0;
            max_value <= '0;
            max_index <= 4'd0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;

                    if (start) begin
                        hidden_idx <= '0;
                        input_idx <= '0;
                        output_idx <= '0;
                        l2_hidden_idx <= '0;
                        argmax_idx <= '0;
                        state <= L1_INIT;
                    end
                end

                L1_INIT: begin
                    acc_reg <= biases_l1[hidden_idx];
                    input_idx <= '0;
                    state <= L1_MAC;
                end

                L1_MAC: begin
                    acc_reg <= mac_acc_out;

                    if (input_idx == INPUT_SIZE - 1) begin
                        state <= L1_STORE;
                    end else begin
                        input_idx <= input_idx + 1'b1;
                    end
                end

                L1_STORE: begin
                    if ((acc_reg >>> SCALE_BITS) < 0) begin
                        hidden_mem[hidden_idx] <= '0;
                    end else begin
                        hidden_mem[hidden_idx] <= acc_reg >>> SCALE_BITS;
                    end

                    if (hidden_idx == HIDDEN_SIZE - 1) begin
                        output_idx <= '0;
                        state <= L2_INIT;
                    end else begin
                        hidden_idx <= hidden_idx + 1'b1;
                        state <= L1_INIT;
                    end
                end

                L2_INIT: begin
                    acc_reg <= biases_l2[output_idx];
                    l2_hidden_idx <= '0;
                    state <= L2_MAC;
                end

                L2_MAC: begin
                    acc_reg <= mac_acc_out;

                    if (l2_hidden_idx == HIDDEN_SIZE - 1) begin
                        state <= L2_STORE;
                    end else begin
                        l2_hidden_idx <= l2_hidden_idx + 1'b1;
                    end
                end

                L2_STORE: begin
                    output_mem[output_idx] <= acc_reg >>> SCALE_BITS;

                    if (output_idx == OUTPUT_SIZE - 1) begin
                        state <= ARGMAX_INIT;
                    end else begin
                        output_idx <= output_idx + 1'b1;
                        state <= L2_INIT;
                    end
                end

                ARGMAX_INIT: begin
                    max_value <= output_mem[0];
                    max_index <= 4'd0;
                    argmax_idx <= 1;
                    state <= ARGMAX_RUN;
                end

                ARGMAX_RUN: begin
                    if (output_mem[argmax_idx] > max_value) begin
                        max_value <= output_mem[argmax_idx];
                        max_index <= argmax_idx[3:0];
                    end

                    if (argmax_idx == OUTPUT_SIZE - 1) begin
                        state <= ARGMAX_DONE;
                    end else begin
                        argmax_idx <= argmax_idx + 1'b1;
                    end
                end

                ARGMAX_DONE: begin
                    prediction <= max_index;
                    state <= DONE_STATE;
                end

                DONE_STATE: begin
                    done <= 1'b1;

                    if (!start) begin
                        state <= IDLE;
                    end
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule