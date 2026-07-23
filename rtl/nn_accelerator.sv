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
    $readmemh("data/mem/input.mem",      input_mem);
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
    DONE
} state_t;

state_t state;


logic [$clog2(INPUT_SIZE)-1:0]  input_idx;
logic [$clog2(HIDDEN_SIZE)-1:0] hidden_idx;
logic [$clog2(OUTPUT_SIZE)-1:0] output_idx;
logic [$clog2(HIDDEN_SIZE)-1:0] l2_hidden_idx;
logic [$clog2(OUTPUT_SIZE)-1:0] argmax_idx;

logic signed [ACC_WIDTH-1:0] acc_reg;
logic signed [ACC_WIDTH-1:0] acc_next;

logic signed [ACC_WIDTH-1:0]    mac_input;
logic signed [WEIGHT_WIDTH-1:0] mac_weight;
logic signed [ACC_WIDTH-1:0]    mac_acc_out;

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
            mac_input = {{(ACC_WIDTH-INPUT_WIDTH){input_mem[input_idx][INPUT_WIDTH-1]}}, input_mem[input_idx]};
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
    

end

endmodule
