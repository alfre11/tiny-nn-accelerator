module mac_unit #(
    parameter int INPUT_WIDTH  = 8,
    parameter int WEIGHT_WIDTH = 8,
    parameter int ACC_WIDTH    = 32
)(
    input  logic signed [INPUT_WIDTH-1:0]  input_value,
    input  logic signed [WEIGHT_WIDTH-1:0] weight_value,
    input  logic signed [ACC_WIDTH-1:0]    acc_in,
    output logic signed [ACC_WIDTH-1:0]    acc_out
);

    localparam int PRODUCT_WIDTH = INPUT_WIDTH + WEIGHT_WIDTH;

    logic signed [PRODUCT_WIDTH-1:0] product;
    logic signed [ACC_WIDTH-1:0]     product_ext;

    assign product = input_value * weight_value;

    assign product_ext = $signed({
        {(ACC_WIDTH - PRODUCT_WIDTH){product[PRODUCT_WIDTH-1]}},
        product
    });

    assign acc_out = acc_in + product_ext;

endmodule
