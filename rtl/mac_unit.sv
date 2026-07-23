module mac_unit #(
    // DATA_WIDTH is 32 so the same MAC can handle:
    //   Layer 1: sign-extended int8 input  * int8 weight
    //   Layer 2: int32 hidden activation   * int8 weight
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
    logic signed [ACC_WIDTH-1:0]     product_acc_width;

    assign product = input_value * weight_value;

    // Convert the product to ACC_WIDTH before adding it to the accumulator.
    //
    // If the full product is wider than ACC_WIDTH, keep the lower ACC_WIDTH bits.
    // This is equivalent to normal two's-complement truncation. For this project,
    // the products are expected to fit in 32 bits, so this preserves the value.
    //
    // If the product is narrower than ACC_WIDTH, explicitly sign-extend it.
    generate
        if (PRODUCT_WIDTH >= ACC_WIDTH) begin : gen_truncate_product
            assign product_acc_width = $signed(product[ACC_WIDTH-1:0]);
        end else begin : gen_sign_extend_product
            assign product_acc_width = $signed({
                {(ACC_WIDTH - PRODUCT_WIDTH){product[PRODUCT_WIDTH-1]}},
                product
            });
        end
    endgenerate

    assign acc_out = acc_in + product_acc_width;

endmodule
