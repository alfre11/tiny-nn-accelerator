module mac_unit #(
    parameter int DATA_WIDTH   = 32,
    parameter int WEIGHT_WIDTH = 8,
    parameter int ACC_WIDTH    = 32
)(
    input  logic signed [DATA_WIDTH-1:0]   input_value,
    input  logic signed [WEIGHT_WIDTH-1:0] weight_value,
    input  logic signed [ACC_WIDTH-1:0]    acc_in,
    output logic signed [ACC_WIDTH-1:0]    acc_out
);

    localparam int PRODUCT_WIDTH = DATA_WIDTH + WEIGHT_WIDTH;

    logic signed [PRODUCT_WIDTH-1:0] product;
    logic signed [PRODUCT_WIDTH-1:0] acc_ext;
    logic signed [PRODUCT_WIDTH-1:0] sum_ext;

    assign product = input_value * weight_value;

    assign acc_ext = {
        {(PRODUCT_WIDTH - ACC_WIDTH){acc_in[ACC_WIDTH-1]}},
        acc_in
    };

    assign sum_ext = acc_ext + product;

    assign acc_out = $signed(sum_ext[ACC_WIDTH-1:0]);

endmodule