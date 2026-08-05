module mac4_unit #(
    parameter int DATA_WIDTH   = 32,
    parameter int WEIGHT_WIDTH = 8,
    parameter int ACC_WIDTH    = 32
)(
    input  logic signed [DATA_WIDTH-1:0]   input0,
    input  logic signed [DATA_WIDTH-1:0]   input1,
    input  logic signed [DATA_WIDTH-1:0]   input2,
    input  logic signed [DATA_WIDTH-1:0]   input3,

    input  logic signed [WEIGHT_WIDTH-1:0] weight0,
    input  logic signed [WEIGHT_WIDTH-1:0] weight1,
    input  logic signed [WEIGHT_WIDTH-1:0] weight2,
    input  logic signed [WEIGHT_WIDTH-1:0] weight3,

    input  logic signed [ACC_WIDTH-1:0]    acc_in,
    output logic signed [ACC_WIDTH-1:0]    acc_out
);

    localparam int PRODUCT_WIDTH = DATA_WIDTH + WEIGHT_WIDTH;
    localparam int SUM_WIDTH =
        (PRODUCT_WIDTH + 2 > ACC_WIDTH) ? PRODUCT_WIDTH + 2 : ACC_WIDTH;

    logic signed [PRODUCT_WIDTH-1:0] product0;
    logic signed [PRODUCT_WIDTH-1:0] product1;
    logic signed [PRODUCT_WIDTH-1:0] product2;
    logic signed [PRODUCT_WIDTH-1:0] product3;

    logic signed [SUM_WIDTH-1:0] product0_ext;
    logic signed [SUM_WIDTH-1:0] product1_ext;
    logic signed [SUM_WIDTH-1:0] product2_ext;
    logic signed [SUM_WIDTH-1:0] product3_ext;
    logic signed [SUM_WIDTH-1:0] acc_ext;
    logic signed [SUM_WIDTH-1:0] pair_sum0;
    logic signed [SUM_WIDTH-1:0] pair_sum1;
    logic signed [SUM_WIDTH-1:0] sum_ext;

    // Four multipliers operate concurrently. The two-level adder tree keeps
    // the combinational path shorter than a serial chain of four additions.
    assign product0 = input0 * weight0;
    assign product1 = input1 * weight1;
    assign product2 = input2 * weight2;
    assign product3 = input3 * weight3;

    assign product0_ext = {{(SUM_WIDTH-PRODUCT_WIDTH){product0[PRODUCT_WIDTH-1]}}, product0};
    assign product1_ext = {{(SUM_WIDTH-PRODUCT_WIDTH){product1[PRODUCT_WIDTH-1]}}, product1};
    assign product2_ext = {{(SUM_WIDTH-PRODUCT_WIDTH){product2[PRODUCT_WIDTH-1]}}, product2};
    assign product3_ext = {{(SUM_WIDTH-PRODUCT_WIDTH){product3[PRODUCT_WIDTH-1]}}, product3};
    assign acc_ext = {{(SUM_WIDTH-ACC_WIDTH){acc_in[ACC_WIDTH-1]}}, acc_in};

    assign pair_sum0 = product0_ext + product1_ext;
    assign pair_sum1 = product2_ext + product3_ext;
    assign sum_ext = acc_ext + pair_sum0 + pair_sum1;

    // Preserve the original MAC's ACC_WIDTH wraparound behavior.
    assign acc_out = sum_ext[ACC_WIDTH-1:0];

endmodule
